/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use ::pxdct::{Pxdct, PxdctError, PxdctExecutor};
use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;

fn pxdct_err_to_py(e: PxdctError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ─── transform kind enum (matches all 16 variants) ───────────────────────────

/// String token → (family, number):
///   "dct"  / "dst"   family
///   type   1 … 8
fn parse_kind(kind: &str) -> PyResult<(&'static str, u8)> {
    let lower = kind.to_ascii_lowercase();
    let (family, num_str) = if let Some(s) = lower.strip_prefix("dct") {
        ("dct", s)
    } else if let Some(s) = lower.strip_prefix("dst") {
        ("dst", s)
    } else {
        return Err(PyValueError::new_err(format!(
            "Unknown transform kind '{kind}'. Expected e.g. 'dct2', 'dst4'."
        )));
    };
    let num: u8 = num_str
        .parse()
        .map_err(|_| PyValueError::new_err(format!("Cannot parse type number from '{kind}'")))?;
    if !(1..=8).contains(&num) {
        return Err(PyValueError::new_err(format!(
            "Type {num} out of range; supported types are 1–8"
        )));
    }
    Ok((family, num))
}

// ─── inner planner (type-erased, heap-allocated) ─────────────────────────────

enum Executor {
    F32(Arc<dyn PxdctExecutor<f32> + Send + Sync>),
    F64(Arc<dyn PxdctExecutor<f64> + Send + Sync>),
}

fn build_executor_f32(
    family: &str,
    ty: u8,
    length: usize,
) -> PyResult<Arc<dyn PxdctExecutor<f32> + Send + Sync>> {
    match (family, ty) {
        ("dct", 1) => Pxdct::make_dct1_f32(length),
        ("dct", 2) => Pxdct::make_dct2_f32(length),
        ("dct", 3) => Pxdct::make_dct3_f32(length),
        ("dct", 4) => Pxdct::make_dct4_f32(length),
        ("dct", 5) => Pxdct::make_dct5_f32(length),
        ("dct", 6) => Pxdct::make_dct6_f32(length),
        ("dct", 7) => Pxdct::make_dct7_f32(length),
        ("dct", 8) => Pxdct::make_dct8_f32(length),
        ("dst", 1) => Pxdct::make_dst1_f32(length),
        ("dst", 2) => Pxdct::make_dst2_f32(length),
        ("dst", 3) => Pxdct::make_dst3_f32(length),
        ("dst", 4) => Pxdct::make_dst4_f32(length),
        ("dst", 5) => Pxdct::make_dst5_f32(length),
        ("dst", 6) => Pxdct::make_dst6_f32(length),
        ("dst", 7) => Pxdct::make_dst7_f32(length),
        ("dst", 8) => Pxdct::make_dst8_f32(length),
        _ => unreachable!(),
    }
    .map_err(pxdct_err_to_py)
}

fn build_executor_f64(
    family: &str,
    ty: u8,
    length: usize,
) -> PyResult<Arc<dyn PxdctExecutor<f64> + Send + Sync>> {
    match (family, ty) {
        ("dct", 1) => Pxdct::make_dct1_f64(length),
        ("dct", 2) => Pxdct::make_dct2_f64(length),
        ("dct", 3) => Pxdct::make_dct3_f64(length),
        ("dct", 4) => Pxdct::make_dct4_f64(length),
        ("dct", 5) => Pxdct::make_dct5_f64(length),
        ("dct", 6) => Pxdct::make_dct6_f64(length),
        ("dct", 7) => Pxdct::make_dct7_f64(length),
        ("dct", 8) => Pxdct::make_dct8_f64(length),
        ("dst", 1) => Pxdct::make_dst1_f64(length),
        ("dst", 2) => Pxdct::make_dst2_f64(length),
        ("dst", 3) => Pxdct::make_dst3_f64(length),
        ("dst", 4) => Pxdct::make_dst4_f64(length),
        ("dst", 5) => Pxdct::make_dst5_f64(length),
        ("dst", 6) => Pxdct::make_dst6_f64(length),
        ("dst", 7) => Pxdct::make_dst7_f64(length),
        ("dst", 8) => Pxdct::make_dst8_f64(length),
        _ => unreachable!(),
    }
    .map_err(pxdct_err_to_py)
}

/// A pre-planned DCT/DST executor.
///
/// Create once, call ``execute`` / ``execute_into`` many times.
/// Thread-safe: the inner executor is ``Arc<… + Send + Sync>``.
///
/// Parameters
/// ----------
/// kind : str
///     Transform family and type, e.g. ``"dct2"``, ``"dst4"``, ``"dct8"``.
/// length : int
///     Number of points.
/// dtype : str, optional
///     ``"f32"`` or ``"f64"`` (default ``"f64"``).
#[pyclass(name = "DctPlan")]
struct DctPlan {
    executor: Executor,
    kind: String,
    length: usize,
    dtype: String,
}

#[pymethods]
impl DctPlan {
    #[new]
    #[pyo3(signature = (kind, length, dtype = "f64"))]
    fn new(kind: &str, length: usize, dtype: &str) -> PyResult<Self> {
        let (family, ty) = parse_kind(kind)?;
        let executor = match dtype {
            "f32" | "float32" => Executor::F32(build_executor_f32(family, ty, length)?),
            "f64" | "float64" => Executor::F64(build_executor_f64(family, ty, length)?),
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown dtype '{other}'. Use 'f32' or 'f64'."
                )));
            }
        };
        Ok(Self {
            executor,
            kind: kind.to_string(),
            length,
            dtype: dtype.to_string(),
        })
    }

    /// Return the transform length this plan was built for.
    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    /// Transform kind string (e.g. ``"dct2"``).
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// Floating-point precision (``"f32"`` or ``"f64"``).
    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }

    fn __repr__(&self) -> String {
        format!(
            "DctPlan(kind='{}', length={}, dtype='{}')",
            self.kind, self.length, self.dtype
        )
    }

    /// Execute the transform **in-place**.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     1-D contiguous array of matching dtype and length.
    ///     Modified in-place.
    ///
    /// Returns
    /// -------
    /// None
    fn execute<'py>(&self, _py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<()> {
        match &self.executor {
            Executor::F32(exec) => {
                let arr = data
                    .cast::<PyArray1<f32>>()
                    .map_err(|_| PyValueError::new_err("Expected a 1-D float32 numpy array"))?;
                self.check_len(arr.len())?;
                let mut buf = unsafe { arr.as_slice_mut() }
                    .map_err(|_| PyRuntimeError::new_err("Array must be C-contiguous"))?;
                exec.execute(&mut buf).map_err(pxdct_err_to_py)
            }
            Executor::F64(exec) => {
                let arr = data
                    .cast::<PyArray1<f64>>()
                    .map_err(|_| PyValueError::new_err("Expected a 1-D float64 numpy array"))?;
                self.check_len(arr.len())?;
                let mut buf = unsafe { arr.as_slice_mut() }
                    .map_err(|_| PyRuntimeError::new_err("Array must be C-contiguous"))?;
                exec.execute(&mut buf).map_err(pxdct_err_to_py)
            }
        }
    }

    #[pyo3(signature = (input, output = None))]
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        output: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.execute_into(py, input, output)
    }

    /// Execute the transform into `output`.
    ///
    /// Parameters
    /// ----------
    /// input : numpy.ndarray
    ///     Read-only source array.
    /// output : numpy.ndarray, optional
    ///     Pre-allocated destination. If omitted, a new array is allocated
    ///     and returned. If provided, it is filled in-place and returned.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     The output array (either the one passed in, or a newly allocated one).
    #[pyo3(signature = (input, output = None))]
    fn execute_into<'py>(
        &self,
        py: Python<'py>,
        input: &Bound<'py, PyAny>,
        output: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match &self.executor {
            Executor::F32(exec) => {
                let src = input.cast::<PyArray1<f32>>().map_err(|_| {
                    PyValueError::new_err("Expected a 1-D float32 numpy array for input")
                })?;
                self.check_len(src.len())?;
                let src_s = unsafe { src.as_slice() }
                    .map_err(|_| PyRuntimeError::new_err("Input must be C-contiguous"))?;

                match output {
                    Some(out) => {
                        let dst = out.cast::<PyArray1<f32>>().map_err(|_| {
                            PyValueError::new_err("Expected a 1-D float32 numpy array for output")
                        })?;
                        self.check_len(dst.len())?;
                        let mut dst_s = unsafe { dst.as_slice_mut() }
                            .map_err(|_| PyRuntimeError::new_err("Output must be C-contiguous"))?;
                        exec.execute_into(src_s, &mut dst_s)
                            .map_err(pxdct_err_to_py)?;
                        Ok(out.clone().into_any())
                    }
                    None => {
                        let mut buf = src_s.to_vec();
                        exec.execute(&mut buf).map_err(pxdct_err_to_py)?;
                        Ok(PyArray1::from_vec(py, buf).into_any())
                    }
                }
            }
            Executor::F64(exec) => {
                let src = input.cast::<PyArray1<f64>>().map_err(|_| {
                    PyValueError::new_err("Expected a 1-D float64 numpy array for input")
                })?;
                self.check_len(src.len())?;
                let src_s = unsafe { src.as_slice() }
                    .map_err(|_| PyRuntimeError::new_err("Input must be C-contiguous"))?;

                match output {
                    Some(out) => {
                        let dst = out.cast::<PyArray1<f64>>().map_err(|_| {
                            PyValueError::new_err("Expected a 1-D float64 numpy array for output")
                        })?;
                        self.check_len(dst.len())?;
                        let mut dst_s = unsafe { dst.as_slice_mut() }
                            .map_err(|_| PyRuntimeError::new_err("Output must be C-contiguous"))?;
                        exec.execute_into(src_s, &mut dst_s)
                            .map_err(pxdct_err_to_py)?;
                        Ok(out.clone().into_any())
                    }
                    None => {
                        let mut buf = src_s.to_vec();
                        exec.execute(&mut buf).map_err(pxdct_err_to_py)?;
                        Ok(PyArray1::from_vec(py, buf).into_any())
                    }
                }
            }
        }
    }
}

impl DctPlan {
    fn check_len(&self, got: usize) -> PyResult<()> {
        if got != self.length {
            Err(PyValueError::new_err(format!(
                "Plan was built for length {}, got array of length {}",
                self.length, got
            )))
        } else {
            Ok(())
        }
    }
}

// ─── 2-D planner ─────────────────────────────────────────────────────────────

/// A pre-planned 2-D DCT executor.
///
/// Applies the 1-D transform independently along rows then columns
/// (or the reverse for the inverse).
///
/// Parameters
/// ----------
/// width_plan : DctPlan
///     Plan for the horizontal (column) dimension.
/// height_plan : DctPlan
///     Plan for the vertical (row) dimension.
#[pyclass(name = "DctPlan2D")]
struct DctPlan2D {
    width: usize,
    height: usize,
    dtype: String,
    executor: Dct2DInner,
}

enum Dct2DInner {
    F32(Arc<dyn ::pxdct::MultidimensionalDctExecutor<f32> + Send + Sync>),
    F64(Arc<dyn ::pxdct::MultidimensionalDctExecutor<f64> + Send + Sync>),
}

#[pymethods]
impl DctPlan2D {
    #[new]
    fn new(width_plan: &DctPlan, height_plan: &DctPlan) -> PyResult<Self> {
        // Both plans must share the same dtype
        if width_plan.dtype != height_plan.dtype {
            return Err(PyValueError::new_err(
                "width_plan and height_plan must have the same dtype",
            ));
        }
        let dtype = width_plan.dtype.clone();
        let width = width_plan.length;
        let height = height_plan.length;

        let executor = match (&width_plan.executor, &height_plan.executor) {
            (Executor::F32(w), Executor::F32(h)) => {
                Dct2DInner::F32(Pxdct::make_2d_dct_f32(w.clone(), h.clone()))
            }
            (Executor::F64(w), Executor::F64(h)) => {
                Dct2DInner::F64(Pxdct::make_2d_dct_f64(w.clone(), h.clone()))
            }
            _ => unreachable!("dtype mismatch already checked"),
        };

        Ok(Self {
            width,
            height,
            dtype,
            executor,
        })
    }

    #[getter]
    fn width(&self) -> usize {
        self.width
    }

    #[getter]
    fn height(&self) -> usize {
        self.height
    }

    #[getter]
    fn dtype(&self) -> &str {
        &self.dtype
    }

    fn __repr__(&self) -> String {
        format!(
            "DctPlan2D(width={}, height={}, dtype='{}')",
            self.width, self.height, self.dtype
        )
    }

    /// Execute the 2-D transform **in-place** on a flat row-major array.
    ///
    /// Parameters
    /// ----------
    /// data : numpy.ndarray
    ///     1-D array of length ``width * height``, row-major (C order).
    fn execute<'py>(&self, _py: Python<'py>, data: &Bound<'py, PyAny>) -> PyResult<()> {
        let expected = self.width * self.height;
        match &self.executor {
            Dct2DInner::F32(exec) => {
                let arr = data
                    .cast::<PyArray1<f32>>()
                    .map_err(|_| PyValueError::new_err("Expected a 1-D float32 numpy array"))?;
                if arr.len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} elements ({}×{}), got {}",
                        expected,
                        self.width,
                        self.height,
                        arr.len()
                    )));
                }
                let mut buf = unsafe { arr.as_slice_mut() }
                    .map_err(|_| PyRuntimeError::new_err("Array must be C-contiguous"))?;
                exec.execute(&mut buf).map_err(pxdct_err_to_py)
            }
            Dct2DInner::F64(exec) => {
                let arr = data
                    .cast::<PyArray1<f64>>()
                    .map_err(|_| PyValueError::new_err("Expected a 1-D float64 numpy array"))?;
                if arr.len() != expected {
                    return Err(PyValueError::new_err(format!(
                        "Expected {} elements ({}×{}), got {}",
                        expected,
                        self.width,
                        self.height,
                        arr.len()
                    )));
                }
                let mut buf = unsafe { arr.as_slice_mut() }
                    .map_err(|_| PyRuntimeError::new_err("Array must be C-contiguous"))?;
                exec.execute(&mut buf).map_err(pxdct_err_to_py)
            }
        }
    }
}

// ─── convenience one-shot functions ──────────────────────────────────────────

/// One-shot DCT/DST.  Allocates a new output array.
///
/// Parameters
/// ----------
/// data : array-like (converted to numpy f64)
/// kind : str  — e.g. ``"dct2"``, ``"dst4"``
/// dtype : str — ``"f32"`` or ``"f64"`` (default ``"f64"``)
///
/// Returns
/// -------
/// numpy.ndarray  (copy, same dtype)
#[pyfunction]
#[pyo3(signature = (data, kind = "dct2", dtype = "f64"))]
fn dct<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    kind: &str,
    dtype: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let (family, ty) = parse_kind(kind)?;
    match dtype {
        "f32" | "float32" => {
            let arr = data.call_method1("astype", ("float32",))?;
            let arr = arr.cast::<PyArray1<f32>>().map_err(|_| {
                PyValueError::new_err("Could not interpret data as 1-D float32 array")
            })?;
            let n = arr.len();
            let exec = build_executor_f32(family, ty, n)?;
            // copy into owned Vec, transform, wrap back
            let mut buf: Vec<f32> = unsafe { arr.as_slice()? }.to_vec();
            exec.execute(&mut buf).map_err(pxdct_err_to_py)?;
            Ok(PyArray1::from_vec(py, buf).into_any())
        }
        "f64" | "float64" => {
            let arr = data.call_method1("astype", ("float64",))?;
            let arr = arr.cast::<PyArray1<f64>>().map_err(|_| {
                PyValueError::new_err("Could not interpret data as 1-D float64 array")
            })?;
            let n = arr.len();
            let exec = build_executor_f64(family, ty, n)?;
            let mut buf: Vec<f64> = unsafe { arr.as_slice()? }.to_vec();
            exec.execute(&mut buf).map_err(pxdct_err_to_py)?;
            Ok(PyArray1::from_vec(py, buf).into_any())
        }
        other => Err(PyValueError::new_err(format!("Unknown dtype '{other}'"))),
    }
}

// ─── module ──────────────────────────────────────────────────────────────────

/// pxdct — fast DCT/DST types I–VIII for Python
///
/// Quick start
/// -----------
/// >>> import numpy as np, pxdct
/// >>> x = np.random.randn(256)
///
/// One-shot (convenient, builds the plan on every call):
/// >>> y = pxdct.dct(x, kind='dct2')
///
/// Reusable plan (build once, call many times — prefer this in loops):
/// >>> plan = pxdct.DctPlan('dct2', 256)
/// >>> out  = np.empty(256)
/// >>> plan.execute_into(x, out)
///
/// 2-D (image processing):
/// >>> wp = pxdct.DctPlan('dct2', 512)
/// >>> hp = pxdct.DctPlan('dct2', 512)
/// >>> plan2d = pxdct.DctPlan2D(wp, hp)
/// >>> img_flat = img.ravel().astype('float64')
/// >>> plan2d.execute(img_flat)
#[pymodule]
fn pxdct(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DctPlan>()?;
    m.add_class::<DctPlan2D>()?;
    m.add_function(wrap_pyfunction!(dct, m)?)?;
    Ok(())
}
