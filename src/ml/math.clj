(ns ml.math)

;; (def safe-math false)

;; (defmacro safe-vector-op [u v]
;;   (when safe-math
;;     `(when-not (= (count ~u) (count ~v))
;;        (throw (Exception. "Vector lengths not equal")))))

;; (defmacro safe-matrix-op [a b]
;;   (when safe-math
;;     `(when-not (= (count (first ~a)) (count ~b))
;;        (throw (Exception. "Matrix dimensions do not match")))))

;; (defn scale [v s]
;;   (mapv * v (repeat s)))

;; (defn add [u v]
;;   (safe-vector-op u v)
;;   (mapv + u v))

;; (defn sub [u v]
;;   (safe-vector-op u v)
;;   (mapv - u v))

;; (defn mul [u v]
;;   (safe-vector-op u v)
;;   (mapv * u v))

;; (defn dot [u v]
;;   (safe-vector-op u v)
;;   (reduce + (mul u v)))

;; (defn mmul [a b]
;;   (safe-matrix-op a b)
;;   (let [b-cols (apply map vector b)]
;;     (for [row a col b-cols]
;;       (mapv #(dot row %) col))))

(defn sigmoid [x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn sigmoid' [a]
  (* a (- 1 a)))

(defn tanh [x]
  (Math/tanh x))

(defn tanh' [a]
  (- 1 (* a a)))

(defn relu [x]
  (max 0 x))

(defn relu' [a]
  (if (neg? a)
    0
    1))

(defn leaky-relu [x]
  (max (* 0.01 x) x))

(defn leaky-relu' [a]
  (if (neg? a)
    0.01
    1))

(defn log-loss [y' y]
  (- (+ (* y (Math/log y'))
        (* (- 1 y) (Math/log (1 - y'))))))

(defn sq-diff [y' y]
  (let [d (- y' y)]
    (* d d)))

(defn log-loss' [y' y]
  (+ (/ (- y) y')
     (/ (- 1 y)
        (- 1 y'))))

(defn sq-diff' [y' y]
  (* 2 (- y' y)))
