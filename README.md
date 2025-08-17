Oudated README

---
# Functions(Rust)
### **mapv(&self, f)** (ndarray)
- Maps new array which closure f is applied from original array
```rust
impl<A, S, D> ArrayBase<S, D>
pub fn mapv<B, F>(&self, mut f: F) -> Array<B, D>
where
    F: FnMut(A) -> B,
    A: Clone,
    S: Data,
    S: RawData<Elem = A>,
    D: Dimension,
```
1. It clones each element of the array (`A: Clone` is required)
2. Passes it by value into closure `f` returning `B` where `f: FnMut(A) -> B`
3. Collects all the resulting `B`'s into new `Array<B, D>` of the same shape (`D: Dimension`).


### **mean(&self)** (ndarray)
- Returns the arithmetic mean $\bar{x}$ of all elements in the array
```rust
impl<A, S, D> ArrayBase<S, D>
pub fn mean(&self) -> Option<A>
where
    A: Clone + FromPrimitive + Add<Output = A> + Div<Output = A> + Zero,
    S: Data<Elem = A>,
    D: Dimension,
```
$$
\bar{x} = \frac{1}{n}\sum^{n}_{i=1}x_{i}
$$


### **fold(&self, init, f)** (rust)
- Folds(accumulates) the result with values where closure f is applied
```rust
impl<'a, I, T> Iterator for Cloned<I>
fn fold<Acc, F>(self, init: Acc, f: F) -> Acc
where
    F: FnMut(Acc, Self::Item) -> Acc,
    T: 'a,
    I: Iterator<Item = &'a T>,
    T: Clone,
```
1. Take the provided initial accumulator value `init` into accumulator `acc`
2. Pull each item out of the iterator in turn (calling `next()` for `iter()` under the hood)
3. For each element `x`, call the closure `f(acc, x)` which returns a new accumulator, and overwrite original `acc`
4. Once the iteration is done, return the final `acc` value.


# Functions(Math)
### Softmax
```rust
fn softmax(z: &Array1<f32>) -> Array1<f32> {
    let max_z = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Array1<f32> = z.mapv(|v| (v - max_z).exp());
    let sum_exps = exps.sum();
    let sum_exps_safe = sum_exps.max(1e-8);

    return exps.mapv(|e| e / sum_exps_safe);
}
```
$$
\begin{align*}
\text{softmax}(z)_{i} &= \frac{\exp(z_{i}-\max_{j}z_{j})}{max(\sum_{j}\exp(z_{j}-\max_{k}z_{k}), \mu)} \\
\text{where} \quad \mu &= 1e^{-8}
\end{align*}
$$


### Layer Normalization(LayerNorm)