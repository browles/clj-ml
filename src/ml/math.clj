(ns ml.math)

(def safe-math false)

(defmacro safe-vector-op [u v]
  (when safe-math
    `(when-not (= (count ~u) (count ~v))
       (throw (Exception. "Vector lengths not equal")))))

(defn scale [v s]
  (mapv * v (repeat s)))

(defn add [u v]
  (safe-vector-op u v)
  (mapv + u v))

(defn sub [u v]
  (safe-vector-op u v)
  (mapv - u v))

(defn mul [u v]
  (safe-vector-op u v)
  (mapv * u v))

(defn dot [u v]
  (safe-vector-op u v)
  (reduce + (mul u v)))

(defn sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))
