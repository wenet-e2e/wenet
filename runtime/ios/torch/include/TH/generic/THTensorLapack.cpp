#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "TH/generic/THTensorLapack.cpp"
#else

/*
Check if self is transpose of a contiguous matrix
*/
static int THTensor_(isTransposedContiguous)(THTensor *self)
{
  return self->stride(0) == 1 && self->stride(1) == self->size(0);
}

/*
Check if self contains any inf or NaN values
*/
static int THTensor_(isFinite)(THTensor *self)
{
  std::atomic<int> finite{1};
  TH_TENSOR_APPLY(scalar_t, self, if (finite && !std::isfinite(*self_data)) {
                        finite = 0;
                        TH_TENSOR_APPLY_hasFinished = 1; break;
                     });
  return finite;
}
/*
If a matrix is a regular contiguous matrix, make sure it is transposed
because this is what we return from Lapack calls.
*/
static void THTensor_(checkTransposed)(THTensor *self)
{
  if(THTensor_(isContiguous)(self))
    THTensor_(transpose)(self, NULL, 0, 1);
  return;
}
/*
newContiguous followed by transpose
Similar to (newContiguous), but checks if the transpose of the matrix
is contiguous and also limited to 2D matrices.
*/
static THTensor *THTensor_(newTransposedContiguous)(THTensor *self)
{
  THTensor *tensor;
  if(THTensor_(isTransposedContiguous)(self))
  {
    THTensor_(retain)(self);
    tensor = self;
  }
  else
  {
    tensor = THTensor_(newContiguous)(self);
    THTensor_(transpose)(tensor, NULL, 0, 1);
  }

  return tensor;
}

/*
Given the result tensor and src tensor, decide if the lapack call should use the
provided result tensor or should allocate a new space to put the result in.

The returned tensor have to be freed by the calling function.

nrows is required, because some lapack calls, require output space smaller than
input space, like underdetermined gels.
*/
static THTensor *THTensor_(checkLapackClone)(THTensor *result, THTensor *src, int nrows)
{
  /* check if user wants to reuse src and if it is correct shape/size */
  if (src == result && THTensor_(isTransposedContiguous)(src) && src->size(1) == nrows)
    THTensor_(retain)(result);
  else if(src == result || result == NULL) /* in this case, user wants reuse of src, but its structure is not OK */
    result = THTensor_(new)();
  else
    THTensor_(retain)(result);
  return result;
}

/*
Same as cloneColumnMajor, but accepts nrows argument, because some lapack calls require
the resulting tensor to be larger than src.
*/
static THTensor *THTensor_(cloneColumnMajorNrows)(THTensor *self, THTensor *src, int nrows)
{
  THTensor *result;
  THTensor *view;

  if (src == NULL)
    src = self;
  result = THTensor_(checkLapackClone)(self, src, nrows);
  if (src == result)
    return result;

  THTensor_(resize2d)(result, src->size(1), nrows);
  THTensor_(checkTransposed)(result);

  if (src->size(0) == nrows) {
    at::Tensor result_wrap = THTensor_wrap(result);
    at::Tensor src_wrap = THTensor_wrap(src);
    at::native::copy_(result_wrap, src_wrap);
  }
  else
  {
    view = THTensor_(newNarrow)(result, 0, 0, src->size(0));
    at::Tensor view_wrap = THTensor_wrap(view);
    at::Tensor src_wrap = THTensor_wrap(src);
    at::native::copy_(view_wrap, src_wrap);
    c10::raw::intrusive_ptr::decref(view);
  }
  return result;
}

/*
Create a clone of src in self column major order for use with Lapack.
If src == self, a new tensor is allocated, in any case, the return tensor should be
freed by calling function.
*/
static THTensor *THTensor_(cloneColumnMajor)(THTensor *self, THTensor *src)
{
  return THTensor_(cloneColumnMajorNrows)(self, src, src->size(0));
}

void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b, THTensor *a)
{
  int free_b = 0;
  // Note that a = NULL is interpreted as a = ra_, and b = NULL as b = rb_.
  if (a == NULL) a = ra_;
  if (b == NULL) b = rb_;
  THArgCheck(a->dim() == 2, 2, "A should have 2 dimensions, but has %d",
      a->dim());
  THArgCheck(!a->is_empty(), 2, "A should not be empty");
  THArgCheck(b->dim() == 1 || b->dim() == 2, 1, "B should have 1 or 2 "
      "dimensions, but has %d", b->dim());
  THArgCheck(!b->is_empty(), 1, "B should not be empty");
  TORCH_CHECK(a->size(0) == b->size(0), "Expected A and b to have same size "
      "at dim 0, but A has ", a->size(0), " rows and B has ", b->size(0), " rows");

  if (b->dim() == 1) {
    b = THTensor_wrap(b).unsqueeze(1).unsafeReleaseTensorImpl();
    free_b = 1;
  }

  int m, n, nrhs, lda, ldb, info, lwork;
  THTensor *work = NULL;
  scalar_t wkopt = 0;

  THTensor *ra__ = NULL;  // working version of A matrix to be passed into lapack GELS
  THTensor *rb__ = NULL;  // working version of B matrix to be passed into lapack GELS

  ra__ = THTensor_(cloneColumnMajor)(ra_, a);

  m = ra__->size(0);
  n = ra__->size(1);
  lda = m;
  ldb = (m > n) ? m : n;

  rb__ = THTensor_(cloneColumnMajorNrows)(rb_, b, ldb);

  nrhs = rb__->size(1);
  info = 0;


  /* get optimal workspace size */
  THLapack_(gels)('N', m, n, nrhs, ra__->data<scalar_t>(), lda,
                  rb__->data<scalar_t>(), ldb,
                  &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THTensor_(newWithSize1d)(lwork);
  THLapack_(gels)('N', m, n, nrhs, ra__->data<scalar_t>(), lda,
                  rb__->data<scalar_t>(), ldb,
                  work->data<scalar_t>(), lwork, &info);

  THLapackCheckWithCleanup("Lapack Error in %s : The %d-th diagonal element of the triangular factor of A is zero",
                           THCleanup(c10::raw::intrusive_ptr::decref(ra__);
                                     c10::raw::intrusive_ptr::decref(rb__);
                                     c10::raw::intrusive_ptr::decref(work);
                                     if (free_b) c10::raw::intrusive_ptr::decref(b);),
                           "gels", info,"");

  /*
   * In the m < n case, if the input b is used as the result (so b == _rb),
   * then rb_ was originally m by nrhs but now should be n by nrhs.
   * This is larger than before, so we need to expose the new rows by resizing.
   */
  if (m < n && b == rb_) {
    THTensor_(resize2d)(rb_, n, nrhs);
  }

  THTensor_(freeCopyTo)(ra__, ra_);
  THTensor_(freeCopyTo)(rb__, rb_);
  c10::raw::intrusive_ptr::decref(work);
  if (free_b) c10::raw::intrusive_ptr::decref(b);
}

#endif
