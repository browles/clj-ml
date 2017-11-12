(ns ml.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as mstats]
            [clojure.pprint :refer [pprint]]
            ;; [ml.math :refer :all]
            [ml.nn :as nn]
            ;; [ml.nn-naive :as nn-naive]
            ;; [ml.stats :refer :all]
            ;; [ml.utils :refer :all]
            ))

;; (defn normalize [x]
;;   (/ (+ 1.0 x) 2.0))

;; (defn anti-normalize [y]
;;   (- (* 2.0 y) 1.0))

;; (def sin-training-set (gen-dataset #(vector (Math/sin %)) #(vector (* 3.14 (r))) 100000))
;; (def sin-test-set (gen-dataset #(vector (Math/sin %)) #(vector (* 3.14 (r))) 10000))

;; (def normalized-training-set (mapv #(vector (first %) (mapv normalize (second %))) sin-training-set))
;; (def normalized-test-set (mapv #(vector (first %) (mapv normalize (second %))) sin-test-set))

;; (def network (nn/gen-network 1 [20] 1))

;; (defonce iterations (nn/training-epoch network 0.01 normalized-training-set))
;; (defonce batch-iterations (nn/batched-training-epoch network 0.01 normalized-training-set 256))

;; (defn network-sin [network item]
;;   (-> (nn/predict network item) first anti-normalize))

;; (defn evaluate [network test-set]
;;   (let [actual (map #(network-sin network (first %)) test-set)
;;         expected (map first (map second test-set))]
;;     (mean-squared-error actual expected)))

;; (defn evaluate-sin-networks []
;;   (let [num-neurons (map #(* 5 %) (range 1 4))
;;         num-epochs 15
;;         networks (mapv #(nn/gen-network 1 [%] 1) num-neurons)
;;         network-iterations (mapv #(nn/batched-training-epoch % 0.01 normalized-training-set 256) networks)
;;         trained-networks (mapv #(nth % num-epochs) network-iterations)]
;;     (prn num-neurons)
;;     (prn (map #(evaluate % normalized-test-set) trained-networks))))

(def X (nn/randn 1 100000 3.14))
(def X-test (nn/randn 1 10000 3.14))

(def Y (m/emap #(Math/sin %) X))
(def Y-test (m/emap #(Math/sin %) X-test))

(def n (nn/new-network 1 [[10 :relu]
                          ;; [3 :relu]
                          [1 :tanh]]
                       :sq-diff))

(defn epoch [network]
  (nn/train network 0.01 X Y))

(defn sgd-epoch [network]
  (nn/sgd network 0.01 X Y))

(defn batched-sgd-epoch [network]
  (nn/batched-sgd network 0.01 256 X Y))

(def trained (iterate epoch n))
(def sgd-trained (iterate sgd-epoch n))
(def batched-sgd-trained (iterate batched-sgd-epoch n))

(defn evaluate [network]
  (nn/compute-cost network X-test Y-test))
