(ns ml.core-matrix-utils
  (:require [clojure.core.matrix :as m]))

(defn randn
  ([n m]
   (m/emap! (fn [_] (Math/random))
            (m/new-matrix n m))))

(defn broadcast-col [v [m-rows n-cols]]
  (when (not= [m-rows 1] (m/shape v))
    (throw (Exception. (format "Incompatible shapes, cannot broadcast-col shape %s to %s"
                               (m/shape v) [m-rows n-cols]))))
  (m/matrix (mapv #(repeat n-cols (first %)) v)))

(defn submatrices [M num-cols]
  (let [[n m] (m/shape M)
        n-matrices (/ m num-cols)]
    (mapv
      (fn [offset]
        (m/matrix (mapv #(take num-cols (drop (* num-cols offset) %)) M)))
      (range 0 (Math/ceil n-matrices)))))
