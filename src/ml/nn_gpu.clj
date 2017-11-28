(ns ml.nn
  (:require [ml.math :refer :all]
            [ml.neanderthal-utils :refer :all]
            [ml.utils :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.cuda :as ncuda]
            [uncomplicate.neanderthal.native :as nnat]
            [uncomplicate.neanderthal.vect-math :as nvm]))

(set! *warn-on-reflection* true)

(def activation-fn->f
  {:sigmoid sigmoid
   :tanh tanh
   :relu relu
   :leaky-relu leaky-relu})

(def activation-fn->dfdz
  {:sigmoid sigmoid'
   :tanh tanh'
   :relu relu'
   :leaky-relu leaky-relu'})

(def loss-fn->f
  {:log-loss log-loss
   :sq-diff sq-diff})

(def loss-fn->dfda
  {:log-loss log-loss'
   :sq-diff sq-diff'})

(defrecord Layer [W b activation-fn])
(defrecord Network [n-inputs n-outputs layers loss-fn])
(defrecord LayerActivation [input output])
(defrecord NetworkActivation [output layer-activations])
(defrecord LayerGradient [dZ dW db])
(defrecord NetworkGradient [next-dZ next-layer layer-gradients])

(defn new-layer [n m activation-fn]
  (Layer.
    (randn n m 10)
    (randn n 1 1)
    activation-fn))

(defn new-network [n-inputs layer-specs loss-fn]
  (reduce (fn [network [n-neurons activation-fn]]
            (let [{:keys [layers n-outputs]} network]
              (assoc network
                :n-outputs n-neurons
                :layers (conj layers (new-layer n-neurons n-outputs activation-fn)))))
          (Network.
            n-inputs
            n-inputs
            []
            loss-fn)
          layer-specs))

(defn activate-layer [input layer]
  (let [{:keys [W b activation-fn]} layer
        [m n] (n-shape W)
        [_ nx-samples] (n-shape input)
        B (broadcast-col b [m nx-samples])]
    (-> (ncore/mm W input)
        (ncore/axpy B)
        (ncore/alter! (get activation-fn->f activation-fn)))))

(defn activate-layer-gpu [input layer nx-samples]
  (ncuda/with-default-engine
    (let [{:keys [W b activation-fn]} layer
          [m n] (n-shape W)]
      (with-release [gpu-input (n-to-gpu! input)
                     gpu-W (n-to-gpu! W)
                     gpu-B (n-to-gpu! (broadcast-col b [m nx-samples]))
                     gpu-W-input (ncore/mm gpu-W gpu-input)]
        (-> gpu-W-input
            (ncore/axpy! gpu-B)
            n-from-gpu!
            (ncore/alter! (get activation-fn->f activation-fn)))))))

(defn compute-dZ-gpu [layer activation output-layer output-dZ]
  (ncuda/with-default-engine
    (let [{:keys [activation-fn]} layer
          W (:W output-layer)
          [m n] (n-shape W)
          [zm zn] (n-shape output-dZ)
          [am an] (n-shape activation)]
      (with-release [gpu-W-T (n-to-gpu! (nnat/fge (ncore/trans W)))
                     gpu-output-dZ (n-to-gpu! output-dZ)
                     gpu-W-T-output-dZ (ncore/mm gpu-W-T gpu-output-dZ)
                     gpu-activation-dz (n-to-gpu! (ncore/alter! (ncore/copy activation)
                                                                (get activation-fn->dfdz activation-fn)))]
        (-> gpu-W-T-output-dZ
            (nvm/mul! gpu-activation-dz)
            n-from-gpu!)))))

(defn compute-output-dZ [A Y dfda dfdz]
  ; TODO: vectorize
  (let [[Ym Yn] (n-shape Y)]
    (nnat/fge Ym Yn
              (mapcat (fn [a-col y-col]
                        (mapv
                          (fn [a y]
                            (* (dfda a y) (dfdz a)))
                          a-col
                          y-col))
                      (ncore/cols A)
                      (ncore/cols Y)))))

(defn compute-gradient-gpu [input dZ nx-samples]
  (ncuda/with-default-engine
    (let [[zm _] (n-shape dZ)]
      (with-release [gpu-input-T (n-to-gpu! (nnat/fge (ncore/trans input)))
                     gpu-dZ (n-to-gpu! dZ)
                     gpu-dZ-input-T (ncore/mm gpu-dZ gpu-input-T)
                     gpu-dW (ncore/scal! (/ 1.0 nx-samples) gpu-dZ-input-T)]
        (LayerGradient.
          dZ
          (n-from-gpu! gpu-dW)
          (->> (nnat/fge zm 1 (mapv ncore/sum (ncore/rows dZ)))
               (ncore/scal! (/ 1.0 nx-samples))))))))

(defn forward-prop [network X]
  (let [[_ nx-samples] (n-shape X)]
    (reduce (fn [network-activation layer]
              (let [{:keys [output layer-activations]} network-activation
                    activation (activate-layer-gpu output layer nx-samples)]
                (assoc network-activation
                  :output activation
                  :layer-activations (conj layer-activations
                                           (LayerActivation. output activation)))))
            (NetworkActivation.
              X
              [])
            (:layers network))))

(defn backward-prop [network network-activation X Y]
  (let [[_ nx-samples] (n-shape X)
        [output-layer & rev-layers] (reverse (:layers network))
        [output-activation & rev-activations] (reverse (:layer-activations network-activation))
        dfda (get loss-fn->dfda (:loss-fn network))
        dfdz (get activation-fn->dfdz (:activation-fn output-layer))
        output-dZ (compute-output-dZ (:output output-activation) Y dfda dfdz)]
    (-> (reduce (fn [network-gradient [layer layer-activation]]
                  (let [{:keys [next-dZ next-layer layer-gradients]} network-gradient
                        {:keys [input output]} layer-activation]
                    (let [dZ (compute-dZ-gpu layer output next-layer next-dZ)
                          layer-gradient (compute-gradient-gpu input dZ nx-samples)]
                      (assoc network-gradient
                        :next-dZ dZ
                        :next-layer layer
                        :layer-gradients (conj layer-gradients layer-gradient)))))
                (NetworkGradient.
                  output-dZ
                  output-layer
                  [(compute-gradient-gpu (:input output-activation) output-dZ nx-samples)])
                (mapv vector rev-layers rev-activations))
        (update :layer-gradients (comp vec reverse)))))

(defn update-weights [network network-gradient learning-rate]
  (assoc network
    :layers (mapv (fn [layer layer-gradient]
                    (let [{:keys [W b]} layer
                          {:keys [dW db]} layer-gradient]
                      (assoc layer
                        :W (ncore/axpy (- learning-rate) dW W)
                        :b (ncore/axpy (- learning-rate) db b))))
                  (:layers network)
                  (:layer-gradients network-gradient))))

(defn train [network learning-rate X Y]
  (let [activations (forward-prop network X)
        gradient (backward-prop network activations X Y)]
    (update-weights network gradient learning-rate)))

(defn sgd [network learning-rate X Y]
  (reduce (fn [n [x y]]
            (train n learning-rate x y))
          network
          (mapv vector (ncore/cols X) (ncore/cols Y))))

(defn batched-sgd [network batch-size learning-rate X Y]
  (reduce (fn [n [X Y]]
            (train n learning-rate X Y))
          network
          (mapv vector (n-submatrices X batch-size) (n-submatrices Y batch-size))))

(defn feed-forward [network X]
  (reduce
    activate-layer
    X
    (:layers network)))

(defn compute-cost [network X Y]
  ; TODO: vectorize
  (let [[Ym Yn] (n-shape Y)
        loss-fn (get loss-fn->f (:loss-fn network))]
    (-> (nnat/fge Ym Yn
                  (mapcat (fn [a-col y-col]
                            (mapv
                              loss-fn
                              a-col
                              y-col))
                          (ncore/cols (feed-forward network X))
                          (ncore/cols Y)))
        ncore/sum
        (/ Yn))))
