(ns ml.utils
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as mstats]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.cuda :as ncuda]
            [uncomplicate.neanderthal.native :as nnat]))

(defn r []
  (Math/random))

(defn r-bit []
  (if (< (r) 0.5)
    0
    1))

(defn chunked-pmap [f n & colls]
  (let [tuples (apply map vector colls)
        chunks (partition-all n tuples)
        process #(mapv (partial apply f) %)]
    (apply concat (pmap process chunks))))

(defn mapvmapv [f colls]
  (mapv #(mapv f %) colls))

(defn gen-dataset [f g n]
  (let [data (vec (repeatedly n g))
        labels (mapv #(apply f %) data)]
    [data labels]))
