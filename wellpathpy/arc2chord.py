import os
import numpy as np

class SurveyError(Exception):
    pass

class SurveyWarning(UserWarning):
    pass

def modulus(theta, modby):
    """
    Calculate the positive cyclic modulus of theta by modby.
    """
    is_scalar = np.ndim(theta) == 0
    theta = np.atleast_1d(theta)
    
    k = (theta / modby).astype(int)
    theta -= k * modby
    theta[theta < 0.0] += modby
    return theta[0] if is_scalar else theta

def quantize(x, fraction=0.5):
    """
    Round a number, or list of numbers, to the closest fraction.

    Examples
    --------
    >>> quantize(1.3, fraction=0.5)
    1.5
    >>> quantize(2.6, fraction=0.5)
    2.5
    >>> quantize(3.0, fraction=0.5)
    3.0
    >>> quantize(4.1, fraction=0.5)
    4.0
    >>> quantize([1.3, 2.6, 3.0 ,4.1], fraction=0.5)
    [1.5, 2.5, 3.0, 4.0]
    """
    
    is_scalar = np.ndim(x) == 0
    x = np.atleast_1d(x)
    
    div = 1.0 / fraction
    qnt = np.round(x * div) / div
    return qnt[0] if is_scalar else qnt

def toUnitDir(degINC, degAZ):
    """
    Convert spherical coordinates to cubic coordinates (NEV), a unit length direction vector
    in a right handed coordinate system.
    """
    dim_inc = np.ndim(degINC)
    is_scalar = dim_inc == 0
    degINC = np.atleast_1d(degINC)
    degAZ = np.atleast_1d(degAZ)
    
    inc = np.deg2rad(degINC)
    az = np.deg2rad(degAZ)
    deltaN = np.sin(inc) * np.cos(az)
    deltaE = np.sin(inc) * np.sin(az)
    deltaV = np.cos(inc)

    nev = np.column_stack((deltaN, deltaE, deltaV))
    return nev[0] if is_scalar else nev

def toSpherical(unit_t):
    """
    Convert a unit direction vector in NEV coordinates to a tuple of Inclination and Azimuth.
    """
    is_1d = np.ndim(unit_t) == 1
    unit_t = np.atleast_2d(unit_t)
    
    inc = np.arctan2(np.sqrt(unit_t[:,0] * unit_t[:,0] + unit_t[:,1] * unit_t[:,1]), unit_t[:,2])
    az = np.arctan2(unit_t[:,1], unit_t[:,0])
    az[az < 0.0] += 2.0 * np.pi # return a result for az such that az is [0.0, 360.0)
    ia = np.column_stack((np.rad2deg(inc), np.rad2deg(az)))
    return ia[0] if is_1d else ia

def slerp(t, u, v, theta=None):
    """
    Spherical linear interpolation.
    Returns a unit vector a fraction, t, between unit vectors u and v.

    t: a float or an array of floats to interpolate.
        u and v: are unit vectors.
    theta: the angle between u and v. If none then the angle is calculated.
    """
    def _sin_over_x(x):
        """
        Numerically stable sin_over_x function.
        """
        mask = 1.0 + (x * x) == 1.0
        x[mask] = 1.0
        x[~mask] = np.sin(x[~mask]) / x[~mask]
        return x
    
    is_scalar = np.ndim(t) == 0
    t = np.atleast_1d(t)[:,None]
    if theta is None:
        theta = 2.0 * np.arctan(np.linalg.norm(v - u) / np.linalg.norm(u + v))
    
    q = 1.0 - t
    d = _sin_over_x(np.atleast_1d(theta))[0]
    l = (_sin_over_x(q * theta) / d) * (q * u)
    r = (_sin_over_x(t * theta) / d) * (t * v)
    w = l + r
    
    return w[0] if is_scalar else w

def S_and_T_survey():
    """
    The test survey from Compendium One.
    """
    P1 = np.array([40.00, 40.00, 700.00]) # position at start (tie-in)
    srv = np.array([[702.55,  5.50,   45.00], 
                    [1964.57, 29.75,  77.05], 
                    [5086.35, 29.75,  77.05], 
                    [9901.68, 120.00, 285.00]])
    return srv, P1

def S_and_T_pos_log():
    """
    The test survey from Compendium One.
    """
    head =      ['INT',    'MD', 'INC',   'AZI',    'N',      'E',     'V', 'DLS'] 
    pos = np.array([[0,  702.55,  5.50,   45.00,  40.00,    40.00,  700.00, 0.00], 
                    [0, 1964.57,  29.75,  77.05, 154.78,   393.64, 1895.35, 2.00],
                    [1, 4250.00,  29.75,  77.05, 408.84,  1498.82, 3879.60, 0.00],
                    [0, 5086.35,  29.75,  77.05, 501.82,  1903.25, 4605.73, 0.00],
                    [1, 8504.11,  80.89, 300.71, 1967.04, 1033.30, 7050.00, 3.00],
                    [1, 8828.04,  90.00, 297.31, 2123.40,  751.22, 7075.71, 3.00],
                    [1, 9151.97,  99.11, 293.92, 2262.88,  460.41, 7050.00, 3.00],
                    [0, 9901.68, 120.00, 285.00, 2500.00, -200.00, 6800.00, 3.00]])
    return head, pos

def Adams_pos_log():
    """
    Survey Data Obtained from Adams and Charrier (1985)
    """
    md = [3000, 3300, 3600, 3900, 5000, 6000, 7000, 8000, 9000, 10000]
    inc = [2.0, 4.0, 8.0, 12.0, 15.0, 16.0, 17.0, 17.0, 17.0, 17.0]
    azi = [28.0, 10.0, 35.0, 25.0, 30.0, 28.0, 50.0, 20.0, 30.0, 25.0]
    n = [0.0, 14.93, 42.35, 87.74, 314.71, 548.48, 764.35, 996.13, 1260.16, 1519.26]
    e = [0.0, 4.28, 18.07, 43.24, 162.77, 292.18, 469.05, 631.34, 754.46, 889.34]
    v = [3000.0, 3299.58, 3597.92, 3893.32, 4962.85, 5926.47, 6886.22, 7844.36, 8800.87, 9757.22]
    return np.column_stack((md, inc, azi, n, e, v))

def arc2chord(t1, t2, arclen):
    """
    Calculates the relative vector between two survey stations,
    given tangents at the ends of the arc and the arc length between
    the tangents.
    Assumes survey values are correct
    and if arrays of values, the arrays must all be the same length.
    """
    is_scalar = np.ndim(arclen) == 0
    arclen = np.atleast_1d(arclen)
    cnt = len(arclen)
    t1 = np.atleast_2d(t1)
    t2 = np.atleast_2d(t2)
    
    t_add = t1 + t2 # add the tangent vectors; a vector that points to the end of the arc from the start
    lsqrd_t_add = np.einsum('ij,ij->i', t_add, t_add) # the length squared of the vector sum... same as: np.sum(t12*t12, axis=1)
    anti_parallel = lsqrd_t_add == 0 # test for anti-parallel tangent vectors, the singuar case
    lsqrd_t_add[anti_parallel] = 1.0 # set so we prevents div-by-zero when unitizing the direction vector
    len_t_add = np.sqrt(lsqrd_t_add) # the length of the addition vector
    norm_t_add = np.divide(t_add, len_t_add[:,None]) # normalize the addition vector to unit vector point to the end of the arc
    
    t_sub = t2 - t1 # subtract the tangent vectors; the chord on a unit circle
    lsqrd_t_sub = np.einsum('ij,ij->i', t_sub, t_sub) # the length squared of the vector subtraction
    len_t_sub = np.sqrt(lsqrd_t_sub) # the length of the subtraction vector
    
    alpha = 2.0 * np.arctan(np.divide(len_t_sub, len_t_add)) # the unoriented angle between the tangent vectors; the arc length on a unit circle
    
    geom_test = len_t_sub < alpha # do the degenerate circle geometry test, the straight hole test
    arc_2_chord = np.ones(cnt) # if degenerte, we are at unity
    arc_2_chord[geom_test] = np.divide(len_t_sub[geom_test], alpha[geom_test]) # where not unity, calc ratio
    
    relative_pos = (arclen * arc_2_chord)[:,None] * norm_t_add # arc-2-chord. For robust numeric evaluation the order of operations here are important
    relative_pos[anti_parallel] = np.array([np.nan, np.nan, np.nan]) # set any singuar cases to nan because they have vanished
    
    return (relative_pos[0], alpha[0]) if is_scalar else (relative_pos, alpha)

def position_log(survey, tie_in, dog_leg_course_length=100, report_raw=False, decimals=None):
    """
    Calculate a position log from a deviation survey and tie-in location.
    survey: a list deviation surveys. [[md_0, inc_0, azi_0], [md_1, inc_1, azi_1],...]
    tie_in: the 3D position of the first survey in survey. [N, E, V] at first survey station
    dog_leg_course_length: a float that is the normalisation length to calculate dogleg
        severity, e.g., 100 -> degrees per 100 feet, 30 -> degrees per 30 meters.
    report_raw: how to report the resulting position log.
    decimals: an interger.  The number of decimals to round the relative positions
        before they are cumsum'ed to the absolute positions.  This limits the precision of the
        values propagated with increasing MD.
    """
    survey = np.array(survey)
    md = survey[:,0]
    if np.all(md[:1] > md[:-1]):
        raise SurveyError('All measured depths must be strictly increasing.')
        
    arclen = md[1:] - md[:-1]
    if np.any(arclen <= 0.0):
        raise SurveyError('All arclens must be GT 0.')
    
    inc = modulus(survey[:,1], 180.0)
    if not np.all(inc == survey[:,1]):
        raise SurveyWarning('One or more inclination values are not GTEQ 0 and LT 180.')
    
    az = modulus(survey[:,2], 360.0)
    if not np.all(az == survey[:,2]):
        raise SurveyWarning('One or more azimuth values are not GTEQ 0 and LT 360.')
    
    tangents = toUnitDir(inc, az)
    rela_pos, alpha = arc2chord(tangents[:-1], tangents[1:], arclen)
    if(not decimals is None):
        rela_pos = np.around(rela_pos, decimals=decimals)
    
    pos = tie_in + np.cumsum(rela_pos, axis=0)
    pos = np.concatenate(([tie_in], pos), axis=0)
    
    if report_raw:
        angle = np.concatenate(([0.0], alpha))
        k = np.concatenate(([0.0], np.divide(alpha, arclen)))
        return np.column_stack((md, tangents, pos, angle, k))
    else:
        dog_leg = np.concatenate(([0.0], (np.rad2deg(alpha) * dog_leg_course_length / arclen)))
        return np.column_stack((md, inc, az, pos, dog_leg))

def inslerpolate(survey, tie_in, step=None, dog_leg_course_length = 100, report_raw=False, decimals=None):
    """
    Interpolate a deviation survey via slerp.
    survey: a list deviation surveys. [[md_0, inc_0, azi_0], [md_1, inc_1, azi_1],...]
    tie_in: the 3D position of the first survey in survey. [N, E, V] at first survey station
    step: the step size to interpolate the survey at, or a list of depths to interpolate.
        If step is None, just calculate the survey position log.
    dog_leg_course_length: a float that is the normalisation length to calculate dogleg
        severity, e.g., 100 -> degrees per 100 feet, 30 -> degrees per 30 meters.
    report_raw: how to report the resulting position.
    decimals: an interger.  The number of decimals to round the relative positions
        before they are cumsum'ed to the absolute positions.  This limits the precision of the
        values propagated with increasing MD.
    """
    survey = np.array(survey)
    if step is None: # no interpolation
        return position_log(survey, tie_in, report_raw=report_raw, decimals=decimals) # so just return the position log
    pos_log = position_log(survey, tie_in, report_raw=True, decimals=decimals)

    mds = pos_log[:,0]
    tangents = pos_log[:,1:4]
    # tie_ins = pos_log[:,4:7]
    angles = pos_log[:,7]
    
    if np.ndim(step) == 0: # interpolate at an equal step size
        quant_ends = quantize([mds[0], mds[-1]], step) # force the end points to be at a step position
        if quant_ends[0] < mds[0]: # make sure we do not start before the first md
            quant_ends[0] += step
        interp_depths = np.arange(quant_ends[0], quant_ends[1] + step, step) # equal md steps to interpolate at
    else:
        interp_depths = np.atleast_1d(step) # user has passed specific interpolation md's
        if np.any(interp_depths < mds[0]):
            raise SurveyError('Some interpolation depths are less than the first depth in the survey.')

    # get the indexes of the begining and ending stations for each interpolated point that the points falls in
    # a md will be GTEQ to the begining and LT the end of the segment
    interp_idx_b = np.searchsorted(mds, interp_depths) # indexes of the would-be interpolated points
    interp_idx_b[interp_idx_b < 1] = 1 # move zero indexes to one so the 'b' indexes are the end of segments
    interp_idx_a = (interp_idx_b - 1) # move the 'a' indexes to the begining of the segments

    t = (interp_depths - mds[interp_idx_a]) / (mds[interp_idx_b] - mds[interp_idx_a]) # fraction in each segment of interp points
    v_0 = tangents[interp_idx_a]
    v_1 = tangents[interp_idx_b]
    ang = angles[interp_idx_b] # angle at the end of the segment. we could calc from v_0 and v_1 but we have it already
    v_i = [slerp(t[i], v_0[i], v_1[i], ang[i]) for i in np.arange(len(interp_depths))] # get the tangents at the interpolated points
    inc_azi = toSpherical(v_i) # get the inc and azi of the interpolated tangents
    srv_itp = np.column_stack((interp_depths, inc_azi)) # [[md_0, inc_0, azi_0], [md_1, inc_1, azi_1],...] surveys at interp points
    
    srv_org_dict = {srv[0]: srv for srv in survey} # a dict of our original surveys keyed by md
    srv_itp_dict = {srv[0]: srv for srv in srv_itp}  # a dict of our interpolated surveys keyed by md
    srv_mrg_dict = srv_org_dict.copy()
    srv_mrg_dict.update(srv_itp_dict) # merge the original and interpolated
    srv_cmb = [srv_mrg_dict[md] for md in sorted(srv_mrg_dict.keys())] # combined list of surveys in md order

    pos_log = position_log(srv_cmb, tie_in, dog_leg_course_length=dog_leg_course_length, report_raw=report_raw, decimals=decimals) # calc the postions of the interpolated points
    pos_log_dict = {pos[0]: pos for pos in pos_log} # dict of pos log keyed by md
    interp_pos_logs = [pos_log_dict[md]  for md in interp_depths] # get the interpolated postion log

    return np.array(interp_pos_logs)

def project(survey, tie_in, to_md, curvature=None, report_raw=False, decimals=None):
    """
    Project a deviation survey to a measured depth beyond the last survey station.
    Returns a deviation survey with the projected survey appended to the end.
    If curvature is None, the curvature from the last arc of the survey is used.
    decimals: an interger.  The number of decimals to round the relative positions
        before they are cumsum'ed to the absolute positions.  This limits the precision of the
        values propagated with increasing MD.
    """
    survey = np.asarray(survey)
    pos_log = position_log(survey, tie_in, report_raw=True, decimals=decimals)[-2:] # grab the last two surveys
    
    mds = pos_log[:,0]
    tangents = pos_log[:,1:4]
    angles = pos_log[:,7]
    
    if curvature is None:
        curvature = angles[1] / (mds[1] - mds[0]) # alpha / course_length
    
    if curvature <= 0: # straight hole
        sta_proj = np.concatenate([[to_md], toSpherical(tangents[1])])
        srv_plus = np.concatenate([survey, [sta_proj]], axis=0)
        return position_log(srv_plus, tie_in, report_raw=report_raw, decimals=decimals)
    
    t = (to_md - mds[1]) / (mds[1] - mds[0])
    ang = curvature * (mds[1] - mds[0])
    v_i = slerp(t, tangents[0], tangents[1], ang)
    sta_proj = np.concatenate([[to_md], toSpherical(v_i)])
    srv_plus = np.concatenate([survey, [sta_proj]], axis=0)
    return position_log(srv_plus, tie_in, report_raw=report_raw, decimals=decimals)

def verticalSection(vs_azimuth):
    """
    Calculates the vertical section of point or set of survey positions.
    http://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf
    Paragraph 9: Applications of Cross-Products to Geometrical Problems, #2
    vs_azimuth: the azimuth, in degrees, of the vertical section.
    """
    def _pee_cross(p):
        """
        http://people.eecs.berkeley.edu/~wkahan/MathH110/Cross.pdf
        p: a column vector.
        """
        return np.matrix([[0.0, -p[2], p[1]], [p[2], 0.0, -p[0]], [-p[1], p[0], 0.0]])
    
    az = toUnitDir(90.0, vs_azimuth)
    #u = np.matrix([0.0, 0.0, 0.0]).T # we don't need this as u is always the origin
    v = np.matrix([0.0, 0.0, 1.0]).T
    w = np.matrix(az).T # vector in horizontal plane pointing in the azimuth direction
    p = _pee_cross(v) * w
    pd = p.T * p
    
    def _f(y):
        """
        y: an array of 3D vectors, the position of the points, from the survey,
        relative to the origin of the survey.
        """
        z = y.T - p * p.T * y.T / pd # for the case of VS this in most cases will probably be more stable 
        z = np.asarray(z.T)
        z[:,2] = 0.0
        return np.dot(z, az)
    return _f

def Adams_test():
    adams = Adams_pos_log()
    print("")
    print(adams)
    pos_log = inslerpolate(adams[:,0:3], adams[0,3:6], decimals=None)
    print("")
    print(pos_log)
    print("")
    print(adams - pos_log[:,:-1])
    print("")
    print(inslerpolate(adams[:,0:3], adams[0,3:6], step=1000, decimals=None))

def main():
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.3f}'.format})

    print("Hello World!")
    print("")

    if True:
        Adams_test()
        return

    srv, tie = np.array(S_and_T_survey())
    _, pos = S_and_T_pos_log()
    pos = np.array(pos)
    pos_srv = pos[pos[:,0]==0][:,1:]
    pos_org = pos[pos[:,0]==1][:,1:]
    print(pos_srv)
    print("")
    pos_clc = inslerpolate(srv, tie)
    print(pos_clc)
    print("")
    print(pos_srv - pos_clc)
    print("")
    print(pos_org)
    print("")
    pos_ipt = inslerpolate(srv, tie, pos_org[:,0])
    print(pos_ipt)
    print("")
    print(pos_org - pos_ipt)

    deviation = np.genfromtxt('./data/deviation.csv', delimiter=',', skip_header=1)
    deviation = np.row_stack(([0,0,0,0,0,0,0], deviation))
    devsrv = deviation[:,0:3]
    print("")
    print(devsrv)
    devtie = np.column_stack([deviation[:,4], deviation[:,5], deviation[:,3]])
    print("")
    print(devtie)
    devpos = inslerpolate(devsrv, devtie[0], dog_leg_course_length=30)
    print("")
    print(devpos)
    print("")
    print(deviation[-4:])
    print("")
    print(Adams_pos_log())


  
if __name__== "__main__":
    main()
