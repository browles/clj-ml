(ns ml.core
  (:require [clojure.pprint :refer [pprint]]
            [ml.math :refer :all]
            [ml.nn :as nn]
            [ml.stats :refer :all]
            [ml.utils :refer :all]))

(defn normalize [x]
  (/ (+ 1.0 x) 2.0))

(defn anti-normalize [y]
  (- (* 2.0 y) 1.0))

(def sin-training-set (gen-dataset #(vector (Math/sin %)) #(vector (* 3.14 (r))) 100000))
(def sin-test-set (gen-dataset #(vector (Math/sin %)) #(vector (* 3.14 (r))) 10000))

(def normalized-training-set (mapv #(vector (first %) (mapv normalize (second %))) sin-training-set))
(def normalized-test-set (mapv #(vector (first %) (mapv normalize (second %))) sin-test-set))

;; (def network (nn/gen-network 1 [20] 1))

;; (defonce iterations (nn/training-epoch network 0.01 normalized-training-set))
;; (defonce batch-iterations (nn/batched-training-epoch network 0.01 normalized-training-set 256))

(defn network-sin [network item]
  (-> (nn/predict network item) first anti-normalize))

(defn evaluate [network test-set]
  (let [actual (map #(network-sin network (first %)) test-set)
        expected (map first (map second test-set))]
    (mean-squared-error actual expected)))

(defn evaluate-sin-networks []
  (let [num-neurons (map #(* 5 %) (range 1 4))
        num-epochs 15
        networks (mapv #(nn/gen-network 1 [%] 1) num-neurons)
        network-iterations (mapv #(nn/batched-training-epoch % 0.01 normalized-training-set 256) networks)
        trained-networks (mapv #(nth % num-epochs) network-iterations)]
    (prn num-neurons)
    (prn (map #(evaluate % normalized-test-set) trained-networks))))
