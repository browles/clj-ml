(ns ml.math)

(defn sigmoid ^double [^double x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(defn sigmoid' ^double [^double a]
  (* a (- 1.0 a)))

(defn tanh ^double [^double x]
  (Math/tanh x))

(defn tanh' ^double [^double a]
  (- 1.0 (* a a)))

(defn relu ^double [^double x]
  (max 0.0 x))

(defn relu' ^double [^double a]
  (if (neg? a)
    0.0
    1.0))

(defn leaky-relu ^double [^double x]
  (max (* 0.01 x) x))

(defn leaky-relu' ^double [^double a]
  (if (neg? a)
    0.01
    1.0))

(defn log-loss ^double [^double y' ^double y]
  (- (+ (* y (Math/log y'))
        (* (- 1.0 y) (Math/log (- 1.0 y'))))))

(defn sq-diff ^double [^double y' ^double y]
  (let [d (- y' y)]
    (* d d)))

(defn log-loss' ^double [^double y' ^double y]
  (+ (/ (- y) y')
     (/ (- 1.0 y)
        (- 1.0 y'))))

(defn sq-diff' ^double [^double y' ^double y]
  (* 2.0 (- y' y)))
