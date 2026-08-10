"""
make_filter.py

Waveform Filters:
- generic generalized least squares filter constructor
- Analytic OF constructor:
  - takes waveform template for filter construction.
  - components are waveform shape and derivative
  - can include baseline term
- PCA constructor - (Planned)
  - uses dominant PCA component(s) found from (multiple)? dateset(s)
    - idea is to physically isolate latent variables by
      - Isolating LGAD laser to single spatial coordinate and peak align: should pickout timing contribution
      - Scanning laser across spatial coordinate and reference align: should pickout amplitude contribution
"""
import numpy as np

def buildGenericFilter(components, unusedCov = None, diagCov = True):

    '''
    components: matrix of nSamples x nComponents
     - 1st Component Expected to be Amplitude
     - 2nd Component Expected to be Time Shift / Derivative
     - 3rd Component Expected to be Baseline Shift
    unusedCov: matrix of unmodeled covariance
     - if None, then use identity
    diagCov: boolean
     - if True, only use diagonal elements of covariance matrix
     - if False, use full covariance matrix
    '''

    components = np.asarray(components)
    if components.ndim != 2:
        raise ValueError("components must have shape (nSamples, nComponents)")

    nSamples = components.shape[0]
    
    if unusedCov is None:
        cov = np.eye(nSamples)
    else:
        cov = np.asarray(unusedCov, dtype=float)
        if diagCov:
            cov = np.diag(np.diag(cov))

    cov_inv = np.linalg.pinv(cov, hermitian=True)

    normal_matrix = components.T @ cov_inv @ components
    filters = np.linalg.pinv(normal_matrix, hermitian=True) @ components.T @ cov_inv
    
    return filters


def fromTemplateFilter(template, template_times, times, unusedCov = None, diagCov = True, baseline = True):

    '''
    template: array of N values of waveform shape
    template_times: array of N values of times in samples (100ps)
     - t = 0 must be waveform maximum
    times: times to extract for filter construction in samples (100ps)
     - t = 0 corresponds to waveform maximum
    unusedCov: matrix of unmodeled covariance
    - if None then use identity
    diagCov:
    - if True, only use diagonal elements of covariance matrix
    - if False, use full covariance matrix
    baseline:
    - if True, includes baseline term in construction
    '''
    
    nPoints = len(times)
    times = np.asarray(times)
    template = np.asarray(template)
    template = template/np.max(template)
    template_times = np.asarray(template_times)
    if template.size != template_times.size:
        raise ValueError("Template incorrect, waveform and time axis not the same size")
    deriv = np.gradient(template, template_times)

    template_interp = np.interp(times, template_times, template, left = np.nan, right = np.nan)
    deriv_interp = np.interp(times, template_times, deriv, left = np.nan, right = np.nan)

    if np.isnan(template_interp).any() or np.isnan(deriv_interp).any():
        raise ValueError("desired samples fall outside of template samples")
    
    component_columns = [template_interp, deriv_interp]

    if baseline:
        component_columns.append(np.ones(times.size))

    components = np.column_stack(component_columns)

    return buildGenericFilter(
        components,
        unusedCov=unusedCov,
        diagCov=diagCov,
    )



    
    
