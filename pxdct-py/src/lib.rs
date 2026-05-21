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

// ─── transform kind ──────────────────────────────────────────────────────────

/// Canonical transform key returned by `parse_kind`.
///
/// DCT/DST types 1–8 → `"dct1"` … `"dst8"`.
/// MDCT / IMDCT      → `"mdct"` / `"imdct"`.
///
/// MDCT and IMDCT have no numeric type suffix, so they are accepted as bare
/// strings (case-insensitive).
fn parse_kind(kind: &str) -> PyResult<&'static str> {
    match kind.to_ascii_lowercase().as_str() {
        // bare aliases accepted for MDCT / IMDCT
        "mdct" => return Ok("mdct"),
        "imdct" => return Ok("imdct"),
        _ => {}
    }

    // DCT/DST 1–8
    let lower = kind.to_ascii_lowercase();
    let (family, num_str) = if let Some(s) = lower.strip_prefix("dct") {
        ("dct", s)
    } else if let Some(s) = lower.strip_prefix("dst") {
        ("dst", s)
    } else {
        return Err(PyValueError::new_err(format!(
            "Unknown transform kind '{kind}'. \
             Expected e.g. 'dct2', 'dst4', 'mdct', 'imdct'."
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

    // Return a 'static str so callers don't need to own a String.
    Ok(match (family, num) {
        ("dct", 1) => "dct1",
        ("dct", 2) => "dct2",
        ("dct", 3) => "dct3",
        ("dct", 4) => "dct4",
        ("dct", 5) => "dct5",
        ("dct", 6) => "dct6",
        ("dct", 7) => "dct7",
        ("dct", 8) => "dct8",
        ("dst", 1) => "dst1",
        ("dst", 2) => "dst2",
        ("dst", 3) => "dst3",
        ("dst", 4) => "dst4",
        ("dst", 5) => "dst5",
        ("dst", 6) => "dst6",
        ("dst", 7) => "dst7",
        ("dst", 8) => "dst8",
        _ => unreachable!(),
    })
}

// ─── executor wrapper ────────────────────────────────────────────────────────

enum Executor {
    F32(Arc<dyn PxdctExecutor<f32> + Send + Sync>),
    F64(Arc<dyn PxdctExecutor<f64> + Send + Sync>),
}

fn build_executor_f32(
    key: &str,
    length: usize,
) -> PyResult<Arc<dyn PxdctExecutor<f32> + Send + Sync>> {
    match key {
        "dct1" => Pxdct::make_dct1_f32(length),
        "dct2" => Pxdct::make_dct2_f32(length),
        "dct3" => Pxdct::make_dct3_f32(length),
        "dct4" => Pxdct::make_dct4_f32(length),
        "dct5" => Pxdct::make_dct5_f32(length),
        "dct6" => Pxdct::make_dct6_f32(length),
        "dct7" => Pxdct::make_dct7_f32(length),
        "dct8" => Pxdct::make_dct8_f32(length),
        "dst1" => Pxdct::make_dst1_f32(length),
        "dst2" => Pxdct::make_dst2_f32(length),
        "dst3" => Pxdct::make_dst3_f32(length),
        "dst4" => Pxdct::make_dst4_f32(length),
        "dst5" => Pxdct::make_dst5_f32(length),
        "dst6" => Pxdct::make_dst6_f32(length),
        "dst7" => Pxdct::make_dst7_f32(length),
        "dst8" => Pxdct::make_dst8_f32(length),
        "mdct" => Pxdct::make_mdct_f32(length),
        "imdct" => Pxdct::make_imdct_f32(length),
        _ => unreachable!(),
    }
    .map_err(pxdct_err_to_py)
}

fn build_executor_f64(
    key: &str,
    length: usize,
) -> PyResult<Arc<dyn PxdctExecutor<f64> + Send + Sync>> {
    match key {
        "dct1" => Pxdct::make_dct1_f64(length),
        "dct2" => Pxdct::make_dct2_f64(length),
        "dct3" => Pxdct::make_dct3_f64(length),
        "dct4" => Pxdct::make_dct4_f64(length),
        "dct5" => Pxdct::make_dct5_f64(length),
        "dct6" => Pxdct::make_dct6_f64(length),
        "dct7" => Pxdct::make_dct7_f64(length),
        "dct8" => Pxdct::make_dct8_f64(length),
        "dst1" => Pxdct::make_dst1_f64(length),
        "dst2" => Pxdct::make_dst2_f64(length),
        "dst3" => Pxdct::make_dst3_f64(length),
        "dst4" => Pxdct::make_dst4_f64(length),
        "dst5" => Pxdct::make_dst5_f64(length),
        "dst6" => Pxdct::make_dst6_f64(length),
        "dst7" => Pxdct::make_dst7_f64(length),
        "dst8" => Pxdct::make_dst8_f64(length),
        "mdct" => Pxdct::make_mdct_f64(length),
        "imdct" => Pxdct::make_imdct_f64(length),
        _ => unreachable!(),
    }
    .map_err(pxdct_err_to_py)
}

/// A pre-planned DCT / DST / MDCT / IMDCT executor.
///
/// Create once, call ``execute`` / ``execute_into`` many times.
/// Thread-safe: the inner executor is ``Arc<… + Send + Sync>``.
///
/// Parameters
/// ----------
/// kind : str
///     Transform family and type, e.g. ``"dct2"``, ``"dst4"``, ``"dct8"``,
///     ``"mdct"``, ``"imdct"``.
///     MDCT / IMDCT require an even *length*.
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
        let key = parse_kind(kind)?;
        let executor = match dtype {
            "f32" | "float32" => Executor::F32(build_executor_f32(key, length)?),
            "f64" | "float64" => Executor::F64(build_executor_f64(key, length)?),
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

    /// Transform kind string (e.g. ``"dct2"`` or ``"mdct"``).
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
        if self.is_out_of_place_only() {
            return Err(PyValueError::new_err(format!(
                "'{}' cannot be executed in-place (input and output sizes differ).                  Use execute_into(input, output) instead.",
                self.kind
            )));
        }
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
        let expected_in = self.input_len();
        let expected_out = self.output_len();

        match &self.executor {
            Executor::F32(exec) => {
                let src = input.cast::<PyArray1<f32>>().map_err(|_| {
                    PyValueError::new_err("Expected a 1-D float32 numpy array for input")
                })?;
                if src.len() != expected_in {
                    return Err(PyValueError::new_err(format!(
                        "Plan expects input length {expected_in}, got {}",
                        src.len()
                    )));
                }
                let src_s = unsafe { src.as_slice() }
                    .map_err(|_| PyRuntimeError::new_err("Input must be C-contiguous"))?;

                match output {
                    Some(out) => {
                        let dst = out.cast::<PyArray1<f32>>().map_err(|_| {
                            PyValueError::new_err("Expected a 1-D float32 numpy array for output")
                        })?;
                        if dst.len() != expected_out {
                            return Err(PyValueError::new_err(format!(
                                "Plan expects output length {expected_out}, got {}",
                                dst.len()
                            )));
                        }
                        let mut dst_s = unsafe { dst.as_slice_mut() }
                            .map_err(|_| PyRuntimeError::new_err("Output must be C-contiguous"))?;
                        exec.execute_into(src_s, &mut dst_s)
                            .map_err(pxdct_err_to_py)?;
                        Ok(out.clone().into_any())
                    }
                    None => {
                        let mut buf = vec![0f32; expected_out];
                        exec.execute_into(src_s, &mut buf)
                            .map_err(pxdct_err_to_py)?;
                        Ok(PyArray1::from_vec(py, buf).into_any())
                    }
                }
            }
            Executor::F64(exec) => {
                let src = input.cast::<PyArray1<f64>>().map_err(|_| {
                    PyValueError::new_err("Expected a 1-D float64 numpy array for input")
                })?;
                if src.len() != expected_in {
                    return Err(PyValueError::new_err(format!(
                        "Plan expects input length {expected_in}, got {}",
                        src.len()
                    )));
                }
                let src_s = unsafe { src.as_slice() }
                    .map_err(|_| PyRuntimeError::new_err("Input must be C-contiguous"))?;

                match output {
                    Some(out) => {
                        let dst = out.cast::<PyArray1<f64>>().map_err(|_| {
                            PyValueError::new_err("Expected a 1-D float64 numpy array for output")
                        })?;
                        if dst.len() != expected_out {
                            return Err(PyValueError::new_err(format!(
                                "Plan expects output length {expected_out}, got {}",
                                dst.len()
                            )));
                        }
                        let mut dst_s = unsafe { dst.as_slice_mut() }
                            .map_err(|_| PyRuntimeError::new_err("Output must be C-contiguous"))?;
                        exec.execute_into(src_s, &mut dst_s)
                            .map_err(pxdct_err_to_py)?;
                        Ok(out.clone().into_any())
                    }
                    None => {
                        let mut buf = vec![0f64; expected_out];
                        exec.execute_into(src_s, &mut buf)
                            .map_err(pxdct_err_to_py)?;
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

    /// Expected input array length (differs from `self.length` for MDCT).
    fn input_len(&self) -> usize {
        match self.kind.to_ascii_lowercase().as_str() {
            "mdct" => self.length * 2, // input block is 2*N
            _ => self.length,
        }
    }

    /// Expected output array length (differs from `self.length` for IMDCT).
    fn output_len(&self) -> usize {
        match self.kind.to_ascii_lowercase().as_str() {
            "imdct" => self.length * 2, // output block is 2*N
            _ => self.length,
        }
    }

    /// Whether the transform is inherently out-of-place (MDCT / IMDCT).
    fn is_out_of_place_only(&self) -> bool {
        matches!(self.kind.to_ascii_lowercase().as_str(), "mdct" | "imdct")
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

/// One-shot DCT / DST / MDCT / IMDCT.  Allocates a new output array.
///
/// Parameters
/// ----------
/// data : array-like (converted to numpy f64)
/// kind : str  — e.g. ``"dct2"``, ``"dst4"``, ``"mdct"``, ``"imdct"``
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
    let key = parse_kind(kind)?;

    // MDCT/IMDCT are inherently out-of-place: input and output lengths differ.
    // For MDCT:  plan(n), input = 2n, output = n.
    // For IMDCT: plan(n), input = n,  output = 2n.
    // The plan length is inferred from the *input* array: n = len for IMDCT,
    // n = len/2 for MDCT.
    let is_mdct = key == "mdct";
    let is_imdct = key == "imdct";

    match dtype {
        "f32" | "float32" => {
            let arr = data.call_method1("astype", ("float32",))?;
            let arr = arr.cast::<PyArray1<f32>>().map_err(|_| {
                PyValueError::new_err("Could not interpret data as 1-D float32 array")
            })?;
            let in_len = arr.len();
            let buf_in: Vec<f32> = unsafe { arr.as_slice()? }.to_vec();

            if is_mdct {
                if in_len % 2 != 0 {
                    return Err(PyValueError::new_err("MDCT input length must be even"));
                }
                let n = in_len / 2;
                let exec = build_executor_f32(key, n)?;
                let mut buf_out = vec![0f32; n];
                exec.execute_into(&buf_in, &mut buf_out)
                    .map_err(pxdct_err_to_py)?;
                Ok(PyArray1::from_vec(py, buf_out).into_any())
            } else if is_imdct {
                let n = in_len;
                let exec = build_executor_f32(key, n)?;
                let mut buf_out = vec![0f32; n * 2];
                exec.execute_into(&buf_in, &mut buf_out)
                    .map_err(pxdct_err_to_py)?;
                Ok(PyArray1::from_vec(py, buf_out).into_any())
            } else {
                let exec = build_executor_f32(key, in_len)?;
                let mut buf = buf_in;
                exec.execute(&mut buf).map_err(pxdct_err_to_py)?;
                Ok(PyArray1::from_vec(py, buf).into_any())
            }
        }
        "f64" | "float64" => {
            let arr = data.call_method1("astype", ("float64",))?;
            let arr = arr.cast::<PyArray1<f64>>().map_err(|_| {
                PyValueError::new_err("Could not interpret data as 1-D float64 array")
            })?;
            let in_len = arr.len();
            let buf_in: Vec<f64> = unsafe { arr.as_slice()? }.to_vec();

            if is_mdct {
                if in_len % 2 != 0 {
                    return Err(PyValueError::new_err("MDCT input length must be even"));
                }
                let n = in_len / 2;
                let exec = build_executor_f64(key, n)?;
                let mut buf_out = vec![0f64; n];
                exec.execute_into(&buf_in, &mut buf_out)
                    .map_err(pxdct_err_to_py)?;
                Ok(PyArray1::from_vec(py, buf_out).into_any())
            } else if is_imdct {
                let n = in_len;
                let exec = build_executor_f64(key, n)?;
                let mut buf_out = vec![0f64; n * 2];
                exec.execute_into(&buf_in, &mut buf_out)
                    .map_err(pxdct_err_to_py)?;
                Ok(PyArray1::from_vec(py, buf_out).into_any())
            } else {
                let exec = build_executor_f64(key, in_len)?;
                let mut buf = buf_in;
                exec.execute(&mut buf).map_err(pxdct_err_to_py)?;
                Ok(PyArray1::from_vec(py, buf).into_any())
            }
        }
        other => Err(PyValueError::new_err(format!("Unknown dtype '{other}'"))),
    }
}

// ─── module ──────────────────────────────────────────────────────────────────

/// pxdct — fast DCT/DST types I–VIII, MDCT, and IMDCT for Python
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
/// MDCT / IMDCT (length must be even):
/// >>> mdct_plan  = pxdct.DctPlan('mdct',  256)
/// >>> imdct_plan = pxdct.DctPlan('imdct', 256)
/// >>> coeffs = mdct_plan(x)
/// >>> x_back = imdct_plan(coeffs)
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
