(ns ml.utils)

(defn r []
  (Math/random))

(defn r-bit []
  (if (< (r) 0.5)
    0
    1))

(defn r-vec [n]
  (vec (take n (repeatedly r))))

(defn r-mat [n m]
  (vec (take n (repeatedly #(r-vec m)))))

(def n-cpu (.availableProcessors (Runtime/getRuntime)))

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
