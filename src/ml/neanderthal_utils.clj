(ns ml.neanderthal-utils
  (:require [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.cuda :as ncuda]
            [uncomplicate.neanderthal.native :as nnat]))

(defn shape [m]
  [(ncore/mrows m) (ncore/ncols m)])

(defn to-gpu! [m]
  (ncore/transfer! m (apply ncuda/cuge (shape m))))

(defn from-gpu! [m]
  (ncore/transfer! m (apply nnat/fge (shape m))))

(defn randn
  [n m]
  (ncore/alter!
    (nnat/fge n m)
    (fn ^double [^double _]
      (Math/random))))

(defn broadcast-col [v [m-rows n-cols]]
  (when (not= [m-rows 1] (shape v))
    (throw (Exception. (format "Incompatible shapes, cannot broadcast-col shape %s to %s"
                               (shape v) [m-rows n-cols]))))
  (ncore/alter!
    (nnat/fge m-rows n-cols)
    (fn ^double [^long i ^long j ^double x]
      (ncore/entry v i 0))))

(defn submatrices [M num-cols]
  ;TODO
  )
