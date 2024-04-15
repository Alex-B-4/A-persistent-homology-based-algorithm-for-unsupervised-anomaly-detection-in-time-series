from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.weighted_rips_complex import WeightedRipsComplex
from gudhi.dtm_rips_complex import DTMRipsComplex
from scipy.spatial import distance_matrix
from sklearn.decomposition import PCA
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import gudhi as gd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
import dionysus as dio
# Use conda install -c conda-forge dionysus to install dionysus without problems



class Cycler:
    """ Build a rips diagram of the data and and provide access to cycles.
        This class wraps much of the functionality in Dionysus, giving it a clean interface and providing a way to access cycles.
        Warning: This has only been tested to work for 1-cycles. 
    """
    
    def __init__(self, order=1):
        self.order = order
        self.barcode = None
        self.cycles = None
        
        self._diagram = None
        self._filtration = None
    
    def fit_Rips(self, data):
        """ Generate Rips filtration and cycles for data.
        """

       # Generate rips filtration 
    
        #################################################################
        # Change by A. Bois
        maxeps = np.shape(data)[1]*(np.max(data)-np.min(data)) # greater than the maximum distance in data
        #################################################################
        
        simplices = dio.fill_rips(data, self.order+1 , maxeps)
        
        self.from_simplices(simplices)
    
    
    def fit_witness(self, data, n_landmarks=50, kind='weak'):
        """ Generate Euclidean Witness complex filtration and cycles for data.
        """
        
        simplices, landmarks = witness_filtration(data, n_landmarks=n_landmarks, kind=kind)
        
        self.from_simplices(simplices)
        return landmarks
    
    
    def fit_weighted_Rips(self,pc,n_points=100, q=100, sampling='MinMax', show=False):
        """ Generate DTM Rips complex filtration and cycles for data.
        """
        
        simplices, P, weights = DTM_Rips_filtration(pc,n_points=n_points,q=q,sampling=sampling, show=show)
        
        self.from_simplices(simplices)
        return P, weights

    def from_simplices(self, simplices):

        if not isinstance(simplices, dio.Filtration):
            simplices = dio.Filtration(simplices)


 
        # import pdb; pdb.set_trace()
        # Add cone point to force homology to finite length; Dionysus only gives out cycles of finite intervals
        spxs = [dio.Simplex([-1])] + [c.join(-1) for c in simplices]
        for spx in spxs:
            spx.data = 1
            simplices.append(spx)

        # Compute persistence diagram
        persistence = dio.homology_persistence(simplices)
        diagrams = dio.init_diagrams(persistence, simplices)

        # Set all the results
        self._filtration = simplices
        self._diagram = diagrams[self.order]
        self._persistence = persistence
        self.barcode = np.array([(d.birth, d.death) for d in self._diagram])

        self._build_cycles()

    def _build_cycles(self):
        """Create cycles from the diagram of order=self.order
        """
        cycles = {}
        
        intervals = sorted(self._diagram, key=lambda d: d.death-d.birth, reverse=True)

        for interval in self._diagram:
            if self._persistence.pair(interval.data) != self._persistence.unpaired:
                cycle_raw = self._persistence[self._persistence.pair(interval.data)]
                
                # Break dionysus iterator representation so it becomes a list
                cycle = [s for s in cycle_raw]
                cycle = self._data_representation_of_cycle(cycle)
                cycles[interval.data] = cycle
        
        self.cycles = cycles

    def _data_representation_of_cycle(self, cycle_raw):
        cycle = np.array([list(self._filtration[s.index]) for s in cycle_raw])    
        return cycle

    def get_cycle(self, interval):
        """Get a cycle for a particular interval. Must be same type returned from `longest_intervals` or entry in `_diagram`.
        """

        return self.cycles[interval.data]
    
    def get_all_cycles(self):
        return list(self.cycles.values())
    
    def longest_intervals(self, n):
        """Return the longest n intervals. For all intervals, just access diagram directly from _diagram.
        """

        intervals = sorted(self._diagram, key=lambda d: d.death-d.birth, reverse=True)
        return intervals[:n]
    
#    def order_vertices(self, cycle):
#        """ Take a cycle and generate an ordered list of vertices.
#            This representation is much more useful for analysis.
#        """
#        ordered_vertices = [cycle[0][0], cycle[0][1]]
#        next_row = 0

        # TODO: how do I make this better? It seems so hacky
#        for _ in cycle[1:]:
#            next_vertex = ordered_vertices[-1]
#            rows, cols = np.where(cycle == next_vertex)
#            which = np.where(rows != next_row)
#            next_row, next_col = rows[which], (cols[which] + 1) % 2

#            ordered_vertices.append(cycle[next_row,next_col][0])
        
#        return ordered_vertices
    
    #################################################################
        # Change by A. Bois
    def find_loop(self, cycle):
        """ Take a cycle and generate an ordered list of vertices.
            This representation is much more useful for analysis.
        """
        if len(cycle) == 0:
            return [],[]
        
        if len(cycle) == 1:
            return [cycle[0][0], cycle[0][1]], [0]
        
        if len(cycle) <= 3:
            ordered_vertices = [cycle[0][0],cycle[0][1]]
            v = cycle[1][0]
            if v in ordered_vertices:
                ordered_vertices.append(cycle[1][1])
            else:
                ordered_vertices.append(v)
            return ordered_vertices, [i for i in range(len(cycle))]
        
        
        i_prev = [0]
        n_steps = 2
        v0 = cycle[0][0]
        v = cycle[0][1]
        ordered_vertices = [v0, v]
        n = len(cycle)
        
        while n_steps < n: # if the cycle has exactly one loop, n steps are needed, 
                           # and we already added 2 vertices so we start at 2
            for i in [j for j in range(n) if j not in i_prev]:
                v2 = cycle[i][0]
                v3 = cycle[i][1]
                if v == v2:
                    ordered_vertices.append(v3)
                    i_prev.append(i)
                    v = v3
                    break
                if v == v3:
                    ordered_vertices.append(v2)
                    i_prev.append(i)
                    v = v2
                    break
            n_steps += 1 # increment anyway to ensure termination
        if ordered_vertices[-1]==ordered_vertices[0]:
            return ordered_vertices[:-1], i_prev
        
        return ordered_vertices, i_prev
    
    def order_vertices(self, cycle):
        if len(cycle) <= 3:
            return self.find_loop(cycle)[0]
        
        loop, idx = self.find_loop(cycle)
        if len(loop) == len(cycle):
            return loop
        
        else:
            remaining_edges = [cycle[i] for i in range(len(cycle)) if i not in idx]
            
            return loop + self.order_vertices(remaining_edges)
        
    #################################################################



def compute_point_cloud(window, subwindow_dim=2, delay=1, show = False):
    cloud = []
    n = len(window)
    for i in range(0, n):
        if i+subwindow_dim*delay <= n:
            cloud.append([window[j] for j in range(i,i+subwindow_dim*delay, delay)])
    
    if show:
        if subwindow_dim==2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))
            ax1.plot(window)
            ax1.set_title('Time series')
            ax2.scatter(np.asarray(cloud)[:,0],np.asarray(cloud)[:,1])
            ax2.set_title('Delay embedding')
            plt.show()
        elif subwindow_dim==3:
            fig = plt.figure(figsize=(15,6))
            ax1 = fig.add_subplot(1,2,1)
            ax1.set_title('Time series')
            ax2 = fig.add_subplot(1,2,2,projection = '3d')
            ax2.set_title('Delay embedding')
            ax1.plot(window)
            ax2.scatter(np.asarray(cloud)[:,0],np.asarray(cloud)[:,1],np.asarray(cloud)[:,2])
            plt.show()
        elif subwindow_dim>3:
            fig = plt.figure(figsize=(15,6))
            ax1 = fig.add_subplot(1,2,1)
            ax1.set_title('Time series')
            ax2 = fig.add_subplot(1,2,2,projection = '3d')
            ax1.plot(window)
            pca = PCA(n_components=3)
            pc = np.array(cloud)
            pca_cloud = pca.fit_transform(pc)
            #pca_cloud = pc # just project to avoid PCA convergence problem
            xs, ys, zs = pca_cloud[:,0], pca_cloud[:,1], pca_cloud[:,2]
            ax2.scatter(xs, ys, zs)
            ax2.set_title('Delay embedding (PCA)')
            plt.show()

    return np.array(cloud)

    

def dist_to_cycle(x, cycle_points):
    d = np.linalg.norm(cycle_points - x, axis=1)
    return np.min(d)

### Heuristics to compute persistence and birth threshold ### 

def _jump_cut(vector : np.ndarray):
    """set the cut to the maximum jump

    Args:
        vector (np.ndarray): _description_
    """
    arr = np.sort(vector)
    jumps = np.diff(arr)
    threshold_idx = np.argmax(jumps)
    return (arr[threshold_idx] + arr[threshold_idx+1])/2

def max_plus_n_jumps_cut(vector : np.ndarray, n=0):
    """set the cut to include n jumps after the largest one

    Args:
        vector (np.ndarray): _description_
    """
    if len(vector)<n:# if less than n points, keep them all
        return -1
    
    arr = sorted(vector, reverse=True)
    jumps = np.diff(arr)*(-1) # np.diff gives arr[i+1]-arr[i]
    max_idx = np.argmax(jumps)
    threshold_idx = min(max_idx+n, len(jumps)-1)      

    return (arr[threshold_idx] + arr[threshold_idx+1])/2
    



def DTM_Rips_filtration(pc, n_points=50, q=100, sampling = 'MinMax', show=False):  
    # q: number of nearest neigbors in DTM
    n = len(pc)
    #D = distance_matrix(pc,pc)
    
    if sampling == 'MinMax':
        P = gd.subsampling.choose_n_farthest_points(pc, nb_points=n_points, starting_point=0)
    
    else:
        # random sampling
        P = gd.pick_n_random_points(points=pc, nb_points=n_points)
        
    P = np.array(P)
    #D_P = D[idxs][:,idxs]  # for subsampling by hand
    D_P = distance_matrix(P,P)
    
    knn = gd.point_cloud.knn.KNearestNeighbors(k=q, return_index=False, return_distance=True,
                                               implementation='sklearn')
    knn.fit(pc)
    knn_P = knn.transform(P)
    dtm = DistanceToMeasure(k=q, q=2, metric='precomputed') 
    weights = dtm.fit(pc).transform(knn_P)
    #print("Average weight: ", np.mean(weights))
    w_rips = WeightedRipsComplex(distance_matrix=D_P, weights=weights)

    simplex_tree = w_rips.create_simplex_tree(max_dimension=2)
    filtration = [s for s in simplex_tree.get_filtration()]
    
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        #ax.scatter(P[:,0],P[:,1],P[:,2])
        pca = PCA(n_components=3)
        pca_P = pca.fit_transform(P)
        ax.scatter(pca_P[:,0],pca_P[:,1],pca_P[:,2])
        ax.set_title('Subsampled Delay embedding (PCA)')
        plt.show()
        diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence = 0.00, persistence_dim_max=False)
        diag = simplex_tree.persistence_intervals_in_dimension(1)
        f, ax = plt.subplots(1,1,figsize=(8,4))
        gd.plot_persistence_diagram(diag, axes=ax)
    return filtration, P, weights

def cycle_anomaly_detection(timeseries, d, tau, q, n_points, n_diag = 2, show = False):  
    # data is a list (use timeseries.values if pandas timeseries)
            
    pc = compute_point_cloud(timeseries, subwindow_dim=d, delay=tau, show = show)
    pc = np.array(pc)
    
    cycler = Cycler()
    
    L, weights = cycler.fit_weighted_Rips(pc,n_points=n_points, q=q, show=show)
    
    diag1 = cycler.barcode


    if len(diag1) == 0:
        cycles = []
    
    else:
        pers_thr = max_plus_n_jumps_cut(diag1[:,1]-diag1[:,0], n=n_diag)
        pers_cut_diag = np.array([[p[0],p[1]] for p in diag1 if p[1]-p[0] >= pers_thr])

        if len(pers_cut_diag) > 1:
            # add minimal birth date so that all cycles can be considered normal
            min_birth = np.min(diag1[:,0])
            birthdates = [min_birth]+list(pers_cut_diag[:,0])
            
            birth_thr = _jump_cut(birthdates)
            
            
            if birth_thr < np.min(pers_cut_diag[:,0]): # if all persistent points are close, we consider them all normal
                cut_diag = pers_cut_diag
            else:
                #due to the DTM filtration, we assume that the most persistent cycle will be normal
                most_persistent_cycle_birth = diag1[np.argmax(diag1[:,1]-diag1[:,0]),0]
                birth_thr = max(most_persistent_cycle_birth, birth_thr)
                cut_diag = pers_cut_diag[pers_cut_diag[:,0] <= birth_thr,:]

        else:
            birth_thr = pers_cut_diag[0,0]
            cut_diag = pers_cut_diag

        n_pers_cycles = len(pers_cut_diag)
        n_main_cycles = len(cut_diag)

    
        # keep the most persistent cycle but exclude those that were born late on the original diagram (anomalies)
        main_intervals = sorted(cycler._diagram,key=lambda d: d.death-d.birth, reverse=True)[:n_pers_cycles]
        main_intervals_without_anomaly = sorted(main_intervals,key=lambda d: d.birth)[:n_main_cycles]
        cycles = [cycler.get_cycle(interval) for interval in main_intervals_without_anomaly]
        #vertex_sets = [cycler.order_vertices(cycle) for cycle in cycles]

    if len(cycles)==0:
        cycles = [[(0,1)]]
        print("No cycles")
        show = False

    
    #else:
    # Distance to main cycles
    dists = [np.inf for x in pc]
    for j in range(len(cycles)):
        main_cycle = cycles[j]
        n = len(main_cycle)
        cycle_indices = list(set([main_cycle[i][0] for i in range(n)]+[main_cycle[i][1] for i in range(n)]))
        cycle_points = L[cycle_indices]
        dists = [min(dist_to_cycle(pc[i],cycle_points), dists[i]) for i in range(len(pc))]


    # For each time t of the timeseries, average (or take the min/max of) the distances to the main cycle over 
    # the averaging_window_size last windows containing t
    m = len(dists)

    profile = [np.mean(dists[max(0,i-d*tau): i+1]) for i in range(m)]+[dists[-1] for i in range(len(timeseries)-m)]
               
    if show:
        main_intervals = sorted(main_intervals,key=lambda d: d.birth)
        anomaly_cycles = [cycler.get_cycle(interval) for interval in main_intervals[n_main_cycles:]]
        colors = plt.cm.rainbow(np.linspace(0,1,1+len(anomaly_cycles)))
        #2D
        if d == 2:
            fig = plt.figure(figsize=(6,6))
            fig.suptitle('Normal cycles (in purprle) and abnormal cycles (in other colors)')
            for cycle in cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v = L[[cycle[i][0],cycle[i][1]]][:,0], L[[cycle[i][0],cycle[i][1]]][:,1]
                    plt.plot(xs_v, ys_v, color = colors[0])
            c=1
            for cycle in anomaly_cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v = L[[cycle[i][0],cycle[i][1]]][:,0], L[[cycle[i][0],cycle[i][1]]][:,1]
                    plt.plot(xs_v, ys_v, color = colors[c], linewidth=3)
                c+=1

            # Plot data
            xs, ys = pc[:,0], pc[:,1]
            plt.scatter(xs, ys, color = 'black')
            plt.show()


        #3D
        if d == 3:
            fig = plt.figure(figsize=(6,6))
            fig.suptitle('Normal cycles (in purprle) and abnormal cycles (in other colors)')
            ax = fig.add_subplot(projection='3d')
            #print(len(cycles), " normal cycles")
            for cycle in cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v, zs_v = L[[cycle[i][0],cycle[i][1]]][:,0], L[[cycle[i][0],cycle[i][1]]][:,1], L[[cycle[i][0],cycle[i][1]]][:,2]
                    ax.plot(xs_v, ys_v, zs_v, color = colors[0], linewidth=3)
                c=1
                for cycle in anomaly_cycles:
                    n = len(cycle)
                    for i in range(n):
                        xs_v, ys_v, zs_v = L[[cycle[i][0],cycle[i][1]]][:,0], L[[cycle[i][0],cycle[i][1]]][:,1], L[[cycle[i][0],cycle[i][1]]][:,2]
                        ax.plot(xs_v, ys_v, zs_v, color = colors[c], linewidth=3)
                    c+=1

            # Plot data
            #xs, ys, zs = pc[:,0], pc[:,1], pc[:,2]
            xs, ys, zs = L[:,0], L[:,1], L[:,2]
            ax.scatter(xs, ys, zs,  c = '0.3')
            plt.show()
        
        if d > 3:
            pca = PCA(n_components=3)
            pca_cloud = pca.fit_transform(L)
            fig = plt.figure(figsize=(6,6))
            fig.suptitle('Normal cycles (in purprle) and abnormal cycles (in other colors)')
            ax = fig.add_subplot(projection='3d')
            #fig2 = go.Figure()
            #palette = getattr(px.colors.qualitative,'Plotly')
            #fig2.add_trace(go.Scatter3d(x = pca_cloud[:,0], y = pca_cloud[:,1], z = pca_cloud[:,2],
            #                       mode='markers', marker=dict(size=2, color = palette[0]),name = 'base signal',showlegend=False))
            #c_plotly=1
            for cycle in cycles:
                n = len(cycle)
                for i in range(n):
                    # print cycle
                    xs_v, ys_v, zs_v = pca_cloud[[cycle[i][0],cycle[i][1]]][:,0], pca_cloud[[cycle[i][0],cycle[i][1]]][:,1],pca_cloud[[cycle[i][0],cycle[i][1]]][:,2]
                    ax.plot(xs_v, ys_v, zs_v, color = colors[0], linewidth=5)
                    #fig2.add_trace(go.Scatter3d(x = xs_v, y = ys_v, z = zs_v,
                    #               mode='lines', marker=dict(size =3, color = palette[c_plotly]),line=dict(width=10),name = 'normal cycle',showlegend=False))
                    
                    #closing the loop
                    xs_v, ys_v, zs_v = pca_cloud[[cycle[i][0],cycle[i][1]]][-1:1,0], pca_cloud[[cycle[i][0],cycle[i][1]]][-1:1,1], pca_cloud[[cycle[i][0],cycle[i][1]]][-1:1,2]
                    ax.plot(xs_v, ys_v, zs_v, color = colors[0], linewidth=5)
                #c_plotly+=1
                    
            c=1
            for cycle in anomaly_cycles:
                n = len(cycle)
                for i in range(n):
                    xs_v, ys_v, zs_v = pca_cloud[[cycle[i][0],cycle[i][1]]][:,0], pca_cloud[[cycle[i][0],cycle[i][1]]][:,1], pca_cloud[[cycle[i][0],cycle[i][1]]][:,2]
                    ax.plot(xs_v, ys_v, zs_v, color = colors[c], linewidth=3)
                    #fig2.add_trace(go.Scatter3d(x = xs_v, y = ys_v, z = zs_v,
                    #               mode='lines',opacity=0.5, marker=dict(size =3, color = palette[c_plotly]),line=dict(width=4),name = 'abnormal cycle',showlegend=False))
                    
                    #closing the loop
                    xs_v, ys_v, zs_v = pca_cloud[[cycle[i][0],cycle[i][1]]][-1:1,0], pca_cloud[[cycle[i][0],cycle[i][1]]][-1:1,1], pca_cloud[[cycle[i][0],cycle[i][1]]][-1:1,2]
                    ax.plot(xs_v, ys_v, zs_v, color = colors[c], linewidth=3)
                c+=1
                #c_plotly+=1
            

            # Plot data
            xs, ys, zs = pca_cloud[:,0], pca_cloud[:,1], pca_cloud[:,2]
            #xs, ys, zs = L[:,0], L[:,1], L[:,2]
            ax.scatter(xs, ys, zs,  c = '0.3')
            plt.show()
            #fig2.show()

        fig = plt.figure(figsize = (20,5))
        fig.suptitle('Time series')
        plt.plot(timeseries)
        plt.show()
        fig = plt.figure(figsize = (20,5))
        fig.suptitle('Anomaly scores')
        plt.plot(profile)
        plt.show()
                   
        
    
    return profile

