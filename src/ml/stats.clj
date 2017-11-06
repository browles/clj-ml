(ns ml.stats)

(defn precision [xs ys]
  (let [tp (filter true? (mapv #(and %1 %2) xs ys))
        fp (filter true? (mapv #(and %1 (not %2)) xs ys))]
    (/ (count tp) (+ (count tp) (count fp)))))

(defn recall [xs ys]
  (let [tp (filter true? (mapv #(and %1 %2) xs ys))
        fn (filter true? (mapv #(and (not %1) %2) xs ys))]
    (/ (count tp) (+ (count tp) (count fn)))))

(defn precision-class [xs ys class]
  (precision (filter #(= class %) xs) (filter #(= class %) ys)))

(defn recall-class [xs ys class]
  (recall (filter #(= class %) xs) (filter #(= class %) ys)))

(defn mean-squared-error [xs ys]
  (->> (map - ys xs)
       (map #(* % %))
       (reduce +)
       (* (/ 1.0 (count xs)))))
