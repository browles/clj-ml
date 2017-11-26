(ns ml.nn
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as mstats]
            [ml.utils :refer :all]))

(set! *warn-on-reflection* true)

(m/set-current-implementation :vectorz)

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
    (cm-randn n m 10)
    (cm-randn n 1 1)
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
  (let [{:keys [W b activation-fn]} layer]
    (->> (m/mmul W input)
         (#(m/add % (cm-broadcast-col b (m/shape %))))
         (m/emap (get activation-fn->f activation-fn)))))

(defn forward-prop [network X]
  (reduce (fn [network-activation layer]
            (let [{:keys [output layer-activations]} network-activation
                  activation (activate-layer output layer)]
              (assoc network-activation
                :output activation
                :layer-activations (conj layer-activations
                                         (LayerActivation. output activation)))))
          (NetworkActivation.
            X
            [])
          (:layers network)))

(defn compute-gradient [dZ m-samples input]
  (LayerGradient.
    (m/div (m/mmul dZ (m/transpose input)) m-samples)
    (m/div (m/matrix (mapv (comp vector mstats/sum) (m/rows dZ))) m-samples)))

(defn backward-prop [network network-activation X Y]
  (let [[_ m-samples] (m/shape X)
        [output-layer & rev-layers] (reverse (:layers network))
        [output-activation & rev-activations] (reverse (:layer-activations network-activation))
        dfda (get loss-fn->dfda (:loss-fn network))
        dfdz (get activation-fn->dfdz (:activation-fn output-layer))
        output-dZ (compute-output-dZ (:output output-activation) Y dfda dfdz)
        output-dZ (m/emap (fn [a y]
                            (* (dfda a y)
                               (dfdz a)))
                          (:output output-activation)
                          Y)]
    (-> (reduce (fn [network-gradient [layer layer-activation]]
                  (let [{:keys [next-dZ next-layer layer-gradients]} network-gradient
                        {:keys [activation-fn]} layer
                        {:keys [input output]} layer-activation]
                    (let [dZ (-> (m/mmul (m/transpose (:W input-layer)) input-dZ)
                                 (m/mul (m/emap (get activation-fn->dfdz activation-fn)
                                                output)))
                          layer-gradient (compute-gradient dZ m-samples input)]
                      (assoc network-gradient
                        :next-dZ dZ
                        :next-layer layer
                        :layer-gradients (conj layer-gradients layer-gradient)))))
                (NetworkGradient.
                  output-dZ
                  output-layer
                  [(compute-gradient output-dZ m-samples (:input output-activation))])
                (mapv vector rev-layers rev-activations))
        (update :layer-gradients (comp vec reverse)))))

(defn update-weights [network network-gradient learning-rate]
  (assoc network
    :layers (mapv (fn [layer layer-gradient]
                    (let [{:keys [W b]} layer
                          {:keys [dW db]} layer-gradient]
                      (assoc layer
                        :W (m/add-scaled W dW (- learning-rate))
                        :bias (m/add-scaled b db (- learning-rate)))))
                  (:layers network)
                  (:layer-gradients network-gradient))))

(defn train [network learning-rate X Y]
  (let [activations (forward-prop network X)
        gradient (backward-prop network activations X Y)]
    (update-weights network gradient learning-rate)))

(defn sgd [network learning-rate X Y]
  (reduce (fn [n [x y]]
            (train n learning-rate (mapv vector x) (mapv vector y)))
          network
          (mapv vector (m/columns X) (m/columns Y))))

(defn batched-sgd [network batch-size learning-rate X Y]
  (reduce (fn [n [X Y]]
            (train n learning-rate X Y))
          network
          (mapv vector (submatrices X batch-size) (submatrices Y batch-size))))

(defn feed-forward [network X]
  (reduce
    activate-layer
    X
    (:layers network)))

(defn compute-cost [network X Y]
  (-> (m/emap (get loss-fn->f (:loss-fn network))
              (feed-forward network X)
              Y)
      mstats/sum
      (m/get-row 0)
      mstats/mean))
