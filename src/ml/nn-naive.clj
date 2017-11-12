(ns ml.nn-naive
  (:require [ml.math :refer :all]
            [ml.utils :refer :all]))

(defn fire-neuron [neuron v]
  (sigmoid (dot neuron v)))

(defn activate-layer [layer input]
  (mapv fire-neuron layer (repeat (conj input 1))))

(defn err [x]
  (* x (- 1.0 x)))

(defn forward-prop [network item]
  (loop [activations [] layers network input item]
    (if (seq layers)
      (let [activation (activate-layer (first layers) input)]
        (recur (conj activations activation) (rest layers) activation))
      activations)))

(defn backward-prop [network network-activations target]
  (let [output (peek network-activations)
        output-error (mapv #(* (- %2 %1) (err %1)) output target)]
    (loop [errors [output-error]
           layers (rest (reverse network)) activations (rest (reverse network-activations))
           next-layer (peek network) next-error output-error]
      (if (seq layers)
        (let [layer (first layers)
              activation (first activations)
              error (vec (map-indexed (fn [i neuron-output]
                                        (* (err neuron-output)
                                           (dot next-error (mapv #(nth % i) next-layer))))
                                      activation))]
          (recur (conj errors error) (rest layers) (rest activations) layer error))
        (vec (reverse errors))))))

(defn update-weights [network network-activations network-errors learning-rate item]
  (loop [new-network [] layers network activations network-activations errors network-errors input item]
    (if (seq layers)
      (let [input-with-bias (conj input 1)
            layer (first layers)
            error (first errors)
            new-layer (mapv #(add %1 (scale input-with-bias (* learning-rate %2))) layer error)]
        (recur (conj new-network new-layer) (rest layers) (rest activations) (rest errors) (first activations)))
      new-network)))

(defn train [network learning-rate [item target]]
  (let [activations (forward-prop network item)
        errors (backward-prop network activations target)]
    (update-weights network activations errors learning-rate item)))

(defn batched-train [network learning-rate batch]
  (let [chunk-size (quot (count batch) n-cpu)
        batch-activations (doall (chunked-pmap #(forward-prop network (first %)) chunk-size batch))
        batch-errors (doall (chunked-pmap #(backward-prop network %1 (second %2)) chunk-size batch-activations batch))]
    (loop [new-network network activations batch-activations errors batch-errors items (map first batch)]
      (if (and (seq activations) (seq errors))
        (recur (update-weights new-network (first activations) (first errors) learning-rate (first items))
               (rest activations) (rest errors) (rest items))
        new-network))))

(defn sgd [network learning-rate training-set]
  (reduce #(train %1 learning-rate %2) network (shuffle training-set)))

(defn batched-sgd [network learning-rate training-set batch-size]
  (reduce #(batched-train %1 learning-rate %2) network (partition-all batch-size (shuffle training-set))))

(defn training-epoch [initial-network learning-rate training-set]
  (iterate #(sgd % learning-rate training-set) initial-network))

(defn batched-training-epoch [initial-network learning-rate training-set batch-size]
  (iterate #(batched-sgd % learning-rate training-set batch-size) initial-network))

(defn gen-network [num-inputs hidden-layers num-outputs]
  (loop [layers [] num-last-inputs num-inputs hidden hidden-layers]
    (if (seq hidden)
      (let [num-current (first hidden)]
        (recur (conj layers (r-mat num-current (inc num-last-inputs))) num-current (rest hidden)))
      (conj layers (r-mat num-outputs (inc num-last-inputs))))))

(defn predict [network item]
  (reduce (fn [v layer]
            (activate-layer layer v))
          item
          network))
