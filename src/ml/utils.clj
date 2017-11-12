(ns ml.utils
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as mstats]
            [ml.utils :refer :all]))

(def n-cpu (.availableProcessors (Runtime/getRuntime)))

(defn r []
  (Math/random))

(defn r-bit []
  (if (< (r) 0.5)
    0
    1))

(defn randn
  ([n m]
   (randn n m 1))
  ([n m s]
   (m/emap! (fn [_]
              (* s (- (Math/random) 0.5)))
            (m/new-matrix n m))))

; Because core.matrix doesn't allow broadcasting columns, apparently.
(defn broadcast-col [v [n-rows n-cols]]
  (when (not= [n-rows 1] (m/shape v))
    (throw (Exception. (format "Incompatible shapes, cannot broadcast-col shape %s to %s"
                               (m/shape v) [n-rows n-rows]))))
  (m/matrix (mapv #(repeat n-cols (first %)) v)))

(defn submatrices [M num-cols]
  (let [[n m] (m/shape M)
        n-matrices (/ m num-cols)]
    (mapv
      (fn [offset]
        (m/matrix (mapv #(take num-cols (drop (* num-cols offset) %)) M)))
      (range 0 (Math/ceil n-matrices)))))

(defn chunked-pmap [f n & colls]
  (let [tuples (apply map vector colls)
        chunks (partition-all n tuples)
        process #(mapv (partial apply f) %)]
    (apply concat (pmap process chunks))))

(defn mapvmapv [f colls]
  (mapv #(mapv f %) colls))

(defn gen-dataset [f g n]
  (let [data (vec (take n (repeatedly g)))
        labels (mapv #(apply f %) data)]
    (mapv vector data labels)))
