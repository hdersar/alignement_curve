"""Example using the ray transform with a custom vector geometry.

We manually build a "circle plus line trajectory" (CLT) geometry by
extracting the vectors from a circular geometry and extending it by
vertical shifts, starting at the initial position.
"""

import numpy as np
import odl
import matplotlib.pyplot as plt

# Reconstruction space: discretized functions on the cube [-20, 20]^3
# with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300],
    dtype='float32')

#phantom = odl.phantom.shepp_logan(reco_space, modified=True)
phantom = odl.phantom.cuboid(reco_space, min_pt=(-1,-1), max_pt=(1,1))
#rot90 = odl.tomo.util.euler_matrix(np.pi / 2)

# Define the domain of the curve
n_proj = 4
dim = 2
traj_space = odl.uniform_discr([0,0], [n_proj, dim], (n_proj, dim))

angle_partition = odl.nonuniform_partition(
    np.linspace(0, 2 * np.pi, n_proj, endpoint=False))
detector_partition = odl.uniform_partition(-30, 30, 256)
circle_geom = odl.tomo.FanFlatGeometry(angle_partition, detector_partition,
                                       src_radius=100, det_radius=0)
ray_trafo_circle = odl.tomo.RayTransform(reco_space, circle_geom)
proj_data_circle = ray_trafo_circle(phantom)



def curve2vec_2d(curve):
    curve = np.asarray(curve)
    rot90 = odl.tomo.util.euler_matrix(np.pi / 2)
    vecs = np.empty((n_proj, 6))
    vecs[:, 0:2] = rot90.dot(curve.T).T
    vecs[:, 2:4] = (0,0)
    
    for i,pt in enumerate(curve):
        normal = -pt/np.linalg.norm(pt)
        det_u = rot90.dot(normal * detector_partition.cell_sides)
        vecs[i, 4:6] = rot90.dot(det_u)
        #vecs[i, 2:4] += 10*vecs[i, 4:6]
    return odl.tomo.ConeVecGeometry(detector_partition.shape, vecs)

'''
def curve2vec_2d(curve):
    curve = np.asarray(curve)
    rot90 = odl.tomo.util.euler_matrix(np.pi / 2)
    vecs = np.empty((n_proj, 6))
    vecs[:, 0:2] = curve
    vecs[:, 2:4] = (0,0)
    
    for i,pt in enumerate(curve):
        normal = -pt/np.linalg.norm(pt)
        det_u = normal * detector_partition.cell_sides
        vecs[i, 4:6] = rot90.dot(det_u)
        #vecs[i, 2:4] += 10*vecs[i, 4:6]
    return odl.tomo.ConeVecGeometry(detector_partition.shape, vecs)

'''
'''
def curve2vec_2d(curve):
    curve = np.asarray(curve)
    rot90 = odl.tomo.util.euler_matrix(np.pi / 2)
    vecs = np.empty((n_proj, 6))
    vecs[:, 0:2] = curve
    
    # Find normal vectors
    from scipy.interpolate import splprep, splev
    tck, u = splprep([curve[:,0], curve[:,1]], s=0)
    normals = splev(u,tck,der=2)
    normals = np.asarray(normals)
    sdd = 100
    for i,pt in enumerate(curve):
        normal = normals[:,i]/np.linalg.norm(normals[:,i])
        #vecs[i, 2:4] = vecs[i, 0:2] + sdd*normal
        vecs[i, 2:4] = (0,0)
        det_u = rot90.dot(normal * detector_partition.cell_sides)
        vecs[i, 4:6] = det_u
        vecs[i, 2:4] += det_u*20
    #print('vecs', vecs[:,2:4])
    return odl.tomo.ConeVecGeometry(detector_partition.shape, vecs)
'''
angles = angle_partition.grid.coord_vectors[0]
curve = np.array([circle_geom.src_position(angle) for angle in angles])
curve_geom = curve2vec_2d(curve)
ray_trafo_curve = odl.tomo.RayTransform(reco_space, curve_geom)
proj_data_curve = ray_trafo_curve(phantom)

curve_dirty = curve.copy()
#curve_dirty[0] += (50, 20)
#curve_dirty[1] += (10, 0)
plt.figure()
plt.scatter(curve_dirty[:,0],curve_dirty[:,1])
#curve_dirty = curve.copy() + 1.0 * np.random.normal(size=curve.shape)
curve_geom_dirty = curve2vec_2d(curve_dirty)
ray_trafo_curve_dirty = odl.tomo.RayTransform(reco_space, curve_geom_dirty)
proj_data_curve_dirty = ray_trafo_curve_dirty(phantom)


class RayTransformFixedTempl(odl.Operator):
    
    def __init__(self, template, data_space):
        ndim = template.space.ndim
        n_proj = data_space.shape[0]
        domain = odl.uniform_discr([0,0], [n_proj, ndim], (n_proj, ndim))
        
        super(RayTransformFixedTempl, self).__init__(
            domain, data_space, linear=False)
                                                             
        self.template = template
        
    def _call(self, curve):
        geom = curve2vec_2d(curve)
        ray_trafo = odl.tomo.RayTransform(self.template.space, geom,
                                          use_cache=False)
        return ray_trafo(self.template)
                

class CurveCostFixedTempl(odl.solvers.Functional):
    
    def __init__(self, data_fit, template):
        ndim = template.space.ndim
        n_proj = data_fit.domain.shape[0]
        domain = odl.uniform_discr([0,0], [n_proj, ndim], (n_proj, ndim))
        super(CurveCostFixedTempl, self).__init__(domain)
        
        self.data_fit = data_fit
        self.ray_trafo = RayTransformFixedTempl(template, data_fit.domain)
        
        grad_templ = odl.Gradient(template.space)(template)
        self.ray_trafos_grad = tuple(
            RayTransformFixedTempl(gt, data_fit.domain) for gt in grad_templ)
        
    def _call(self, curve):
        proj = self.ray_trafo(curve)
        return self.data_fit(proj)

    @property
    def gradient(self):
        func = self
        det_space = odl.uniform_discr_frompartition(
            self.data_fit.domain.partition.byaxis[1:])
        det_one = det_space.one()
        #print(det_one)
        class CurveCostFixedTemplGrad(odl.Operator):
            
            def __init__(self):
                super(CurveCostFixedTemplGrad, self).__init__(
                    func.domain, func.domain)
            
            def _call(self, curve):
                proj = func.ray_trafo(curve)
                proj_grad = [trafo(curve) for trafo in func.ray_trafos_grad]
                data_fit_grad = func.data_fit.gradient(proj)
                #proj_grad[1].show()
                integrands = [data_fit_grad * pg for pg in proj_grad]
                integrand_arrs = [np.asarray(integr) for integr in integrands]
                
                curve_grad = np.asarray(func.domain.element())
                rot90 = odl.tomo.util.euler_matrix(np.pi / 2)
                geom = curve2vec_2d(curve)
                for i, (integr_0, integr_1) in enumerate(zip(*integrand_arrs)):
                    #print(integr_0[50:-50])
                    curve_grad[i, 0] = det_one.inner(det_space.element(integr_0))
                    curve_grad[i, 1] = det_one.inner(det_space.element(integr_1))
                    u = geom.det_axis(i)
                    #print(curve_grad[i, 0])
                    #if i == 0:
                    #    print('i', i, 'u', u)
                    #v /= np.linalg.norm(v)
                    #u /= np.linalg.norm(u)
                    v = rot90.dot(u)
                    #basis_mat = np.array([-v,u])
                    #basis_mat = np.eye(2)
                    #print(curve_grad[i,:])
                    #curve_grad[i,:] = basis_mat.dot(np.asarray(curve_grad[i,:]))
                    #print(u.dot(np.asarray(curve_grad[i,0])))
                    #tmp = curve_grad[i,:].copy()
                    #curve_grad[i,0] = u.dot(np.asarray(tmp))
                    #curve_grad[i,1] = v.dot(np.asarray(tmp))
                    #print(basis_mat)
                    #curve_grad[i,:] = u * np.array(curve_grad[i,:])
                    #fac = np.linalg.det([v,u])
                    fac = 1
                    curve_grad[i,:] *= fac
                # TODO: find out why we need to take negative
                return curve_grad
        
        return CurveCostFixedTemplGrad()

class CurveCostFixedTemplNum(CurveCostFixedTempl):
    
    @property
    def gradient(self):
        step = 2
        return odl.solvers.NumericalGradient(self, step=step)
        


# %%

data_space = ray_trafo_circle.range
data_fit = odl.solvers.L2NormSquared(data_space).translated(proj_data_curve)
cost = CurveCostFixedTempl(data_fit, phantom)
cost_num = CurveCostFixedTemplNum(data_fit, phantom)

n = 25
xlen = 2 * float(n)
ylen = 2 * float(n)
xs = np.linspace(-xlen / 2, xlen / 2, n)
ys = np.linspace(-ylen / 2, ylen / 2, n)
spc = odl.uniform_discr([xs[0], ys[0]], [xs[-1], ys[-1]], (n, n))
shift = np.zeros((curve.shape))

values = np.empty((n, n))
idx = 1
for i, xi in enumerate(xs):
    for j, yj in enumerate(ys):
        shift[idx] = (xi, yj)
        values[i, j] = cost(curve + shift)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(xs, ys)
Z = values.reshape(X.shape)
ax.plot_surface(X, Y, Z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('cost')
#spc.element(values).show()
m = np.unravel_index(values.argmin(),values.shape)
print('min at :', (xs[m[0]], ys[m[1]]))


grads = np.empty((n, n, n_proj, 2))
grads_num = np.empty((n, n, n_proj, 2))
shift = np.zeros((curve.shape))
for i, xi in enumerate(xs):
    for j, yj in enumerate(ys):
        shift[idx] = (xi, yj)
        grads[i, j] = cost.gradient(curve + shift)
        grads_num[i, j] = cost_num.gradient(curve + shift)


fig, ax = plt.subplots()
ax.quiver(xs, ys, -grads[..., idx, 0].T, -grads[..., idx, 1].T)
fig, ax = plt.subplots()
ax.quiver(xs, ys, -grads_num[..., idx, 0].T, -grads_num[..., idx, 1].T)


# %%
"""
fig, curve_ax = plt.subplots()
it = 0
show_step = 1
clear_step = 10

def show_curve(curve):
    global it
    if it % clear_step == 0:
        curve_ax.clear()
    if it % show_step == 0:
        curve = curve.asarray()
        curve_ax.scatter(curve[:, 0], curve[:, 1], s=4)
    
    it += 1
"""

data_space = ray_trafo_circle.range
data_fit = odl.solvers.L2NormSquared(data_space).translated(proj_data_curve)
cost_func = CurveCostFixedTempl(data_fit, phantom)
cost_func_num = CurveCostFixedTemplNum(data_fit, phantom)
curve0 = cost_func.domain.element(curve_dirty.copy())
step = 0.01
maxiter = 1000

store = []
def get_point(curve):
    return curve.asarray()[0]
def asarray(curve):
    return curve.asarray()


callback = odl.solvers.CallbackStore(store) * asarray
# callback = odl.solvers.CallbackApply(show_curve)

odl.solvers.steepest_descent(cost_func, curve0, line_search=step,
                             maxiter=maxiter, callback=callback)


curve0 = np.asarray(curve0)
plt.figure()
plt.scatter(curve0[:,0],curve0[:,1])

pt_index = 0
fig1, ax1 = plt.subplots()
ax1.plot([x[pt_index, 0] for x in store])
ax1.plot([curve[pt_index, 0]] * len(store))

fig2, ax2 = plt.subplots()
ax2.plot([x[pt_index, 1] for x in store])
ax2.plot([curve[pt_index, 1]] * len(store))


#%%
pt_index = 4
fig1, ax1 = plt.subplots()
ax1.plot([x[pt_index, 0] for x in store])
ax1.plot([curve[pt_index, 0]] * len(store))

fig2, ax2 = plt.subplots()
ax2.plot([x[pt_index, 1] for x in store])
ax2.plot([curve[pt_index, 1]] * len(store))
