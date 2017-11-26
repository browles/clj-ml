(ns ml.utils
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as mstats]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.cuda :as ncuda]
            [uncomplicate.neanderthal.native :as nnat]))

(def n-cpu (.availableProcessors (Runtime/getRuntime)))

(defn r []
  (Math/random))

(defn r-bit []
  (if (< (r) 0.5)
    0
    1))

(defn cm-randn
  ([n m s]
   (m/emap! (fn [_]
              (* s (- (Math/random) 0.5)))
            (m/new-matrix n m))))

(defn n-randn
  ([n m]
   (n-randn n m 1))
  ([n m s]
   (ncore/alter!
     (nnat/fge n m)
     (fn ^double [^double _]
       (* s (- (Math/random) 0.5))))))

; core.matrix utilities
; Because core.matrix doesn't allow broadcasting columns, apparently.
(defn cm-broadcast-col [v [m-rows n-cols]]
  (when (not= [m-rows 1] (m/shape v))
    (throw (Exception. (format "Incompatible shapes, cannot broadcast-col shape %s to %s"
                               (m/shape v) [m-rows n-cols]))))
  (m/matrix (mapv #(repeat n-cols (first %)) v)))

(defn cm-submatrices [M num-cols]
  (let [[n m] (m/shape M)
        n-matrices (/ m num-cols)]
    (mapv
      (fn [offset]
        (m/matrix (mapv #(take num-cols (drop (* num-cols offset) %)) M)))
      (range 0 (Math/ceil n-matrices)))))

; neaderthal utilities
(defn n-shape [m]
  [(ncore/mrows m) (ncore/ncols m)])

(defn n-to-gpu! [m]
  (ncore/transfer! m (apply ncuda/cuge (n-shape m))))

(defn n-from-gpu! [m]
  (ncore/transfer! m (apply nnat/fge (n-shape m))))

(defn n-broadcast-col [v [m-rows n-cols]]
  (when (not= [m-rows 1] (n-shape v))
    (throw (Exception. (format "Incompatible shapes, cannot broadcast-col shape %s to %s"
                               (n-shape v) [m-rows n-cols]))))
  (ncore/alter!
    (nnat/fge m-rows n-cols)
    (fn ^double [^long i ^long j ^double x]
      (ncore/entry v i 0))))

(defn n-submatrices [M num-cols])

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
