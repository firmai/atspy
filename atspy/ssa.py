import numpy as np
import pandas as pd
from numpy import matrix as m
from scipy import linalg
try:
    import seaborn
except:
    pass
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 11, 4

class mySSA(object):
    '''Singular Spectrum Analysis object'''
    def __init__(self, time_series):
        
        self.ts = pd.DataFrame(time_series)
        self.ts_name = self.ts.columns.tolist()[0]
        if self.ts_name==0:
            self.ts_name = 'ts'
        self.ts_v = self.ts.values
        self.ts_N = self.ts.shape[0]
        self.freq = self.ts.index.inferred_freq
    
    # @staticmethod
    # def _printer(name, *args):
    #     '''Helper function to print messages neatly'''
    #     print('-'*40)
    #     print(name+':')
    #     for msg in args:
    #         print(msg)  
    
    @staticmethod
    def _dot(x,y):
        '''Alternative formulation of dot product to allow missing values in arrays/matrices'''
        pass
    
    @staticmethod
    def get_contributions(X=None, s=None, plot=True):
        '''Calculate the relative contribution of each of the singular values'''
        lambdas = np.power(s,2)
        frob_norm = np.linalg.norm(X)
        ret = pd.DataFrame(lambdas/(frob_norm**2), columns=['Contribution'])
        ret['Contribution'] = ret.Contribution.round(4)
        if plot:
            ax = ret[ret.Contribution!=0].plot.bar(legend=False)
            ax.set_xlabel("Lambda_i")
            ax.set_title('Non-zero contributions of Lambda_i')
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])
            return ax
        return ret[ret.Contribution>0]
    
    @staticmethod
    def diagonal_averaging(hankel_matrix):
        '''Performs anti-diagonal averaging from given hankel matrix
        Returns: Pandas DataFrame object containing the reconstructed series'''
        mat = m(hankel_matrix)
        L, K = mat.shape
        L_star, K_star = min(L,K), max(L,K)
        new = np.zeros((L,K))
        if L > K:
            mat = mat.T
        ret = []
        
        #Diagonal Averaging
        for k in range(1-K_star, L_star):
            mask = np.eye(K_star, k=k, dtype='bool')[::-1][:L_star,:]
            mask_n = sum(sum(mask))
            ma = np.ma.masked_array(mat.A, mask=1-mask)
            ret+=[ma.sum()/mask_n]
        
        return pd.DataFrame(ret).rename(columns={0:'Reconstruction'})
        
    def view_time_series(self):
        '''Plot the time series'''
        self.ts.plot(title='Original Time Series')
        
    def embed(self, embedding_dimension=None, suspected_frequency=None, verbose=False, return_df=False):
        '''Embed the time series with embedding_dimension window size.
        Optional: suspected_frequency changes embedding_dimension such that it is divisible by suspected frequency'''
        if not embedding_dimension:
            self.embedding_dimension = self.ts_N//2
        else:
            self.embedding_dimension = embedding_dimension
        if suspected_frequency:
            self.suspected_frequency = suspected_frequency
            self.embedding_dimension = (self.embedding_dimension//self.suspected_frequency)*self.suspected_frequency
    
        self.K = self.ts_N-self.embedding_dimension+1
        self.X = m(linalg.hankel(self.ts, np.zeros(self.embedding_dimension))).T[:,:self.K]
        self.X_df = pd.DataFrame(self.X)
        self.X_complete = self.X_df.dropna(axis=1)
        self.X_com = m(self.X_complete.values)
        self.X_missing = self.X_df.drop(self.X_complete.columns, axis=1)
        self.X_miss = m(self.X_missing.values)
        self.trajectory_dimentions = self.X_df.shape
        self.complete_dimensions = self.X_complete.shape
        self.missing_dimensions = self.X_missing.shape
        self.no_missing = self.missing_dimensions[1]==0
            
        # if verbose:
        #     msg1 = 'Embedding dimension\t:  {}\nTrajectory dimensions\t: {}'
        #     msg2 = 'Complete dimension\t: {}\nMissing dimension     \t: {}'
        #     msg1 = msg1.format(self.embedding_dimension, self.trajectory_dimentions)
        #     msg2 = msg2.format(self.complete_dimensions, self.missing_dimensions)
        #     self._printer('EMBEDDING SUMMARY', msg1, msg2)
        
        if return_df:
            return self.X_df
        
    def decompose(self, verbose=False):
        '''Perform the Singular Value Decomposition and identify the rank of the embedding subspace
        Characteristic of projection: the proportion of variance captured in the subspace'''
        X = self.X_com
        self.S = X*X.T
        self.U, self.s, self.V = linalg.svd(self.S)
        self.U, self.s, self.V = m(self.U), np.sqrt(self.s), m(self.V)
        self.d = np.linalg.matrix_rank(X)
        Vs, Xs, Ys, Zs = {}, {}, {}, {}
        for i in range(self.d):
            Zs[i] = self.s[i]*self.V[:,i]
            Vs[i] = X.T*(self.U[:,i]/self.s[i])
            Ys[i] = self.s[i]*self.U[:,i]
            Xs[i] = Ys[i]*(m(Vs[i]).T)
        self.Vs, self.Xs = Vs, Xs
        self.s_contributions = self.get_contributions(X, self.s, False)
        self.r = len(self.s_contributions[self.s_contributions>0])
        self.r_characteristic = round((self.s[:self.r]**2).sum()/(self.s**2).sum(),4)
        self.orthonormal_base = {i:self.U[:,i] for i in range(self.r)}
        
        # if verbose:
        #     msg1 = 'Rank of trajectory\t\t: {}\nDimension of projection space\t: {}'
        #     msg1 = msg1.format(self.d, self.r)
        #     msg2 = 'Characteristic of projection\t: {}'.format(self.r_characteristic)
        #     self._printer('DECOMPOSITION SUMMARY', msg1, msg2)
    
    def view_s_contributions(self, adjust_scale=False, cumulative=False, return_df=False):
        '''View the contribution to variance of each singular value and its corresponding signal'''
        contribs = self.s_contributions.copy()
        contribs = contribs[contribs.Contribution!=0]
        if cumulative:
            contribs['Contribution'] = contribs.Contribution.cumsum()
        if adjust_scale:
            contribs = (1/contribs).max()*1.1-(1/contribs)
        ax = contribs.plot.bar(legend=False)
        ax.set_xlabel("Singular_i")
        ax.set_title('Non-zero{} contribution of Singular_i {}'.\
                     format(' cumulative' if cumulative else '', '(scaled)' if adjust_scale else ''))
        if adjust_scale:
            ax.axes.get_yaxis().set_visible(False)
        vals = ax.get_yticks()
        ax.set_yticklabels(['{:3.0f}%'.format(x*100) for x in vals])
        if return_df:
            return contribs
    
    @classmethod
    def view_reconstruction(cls, *hankel, names=None, return_df=False, plot=True, symmetric_plots=False):
        '''Visualise the reconstruction of the hankel matrix/matrices passed to *hankel'''
        hankel_mat = None
        for han in hankel:
            if isinstance(hankel_mat,m):
                hankel_mat = hankel_mat + han
            else: 
                hankel_mat = han.copy()
        hankel_full = cls.diagonal_averaging(hankel_mat)
        title = 'Reconstruction of signal'
        if names or names==0: 
            title += ' associated with singular value{}: {}'
            title = title.format('' if len(str(names))==1 else 's', names)
        if plot:
            ax = hankel_full.plot(legend=False, title=title)
            if symmetric_plots:
                velocity = hankel_full.abs().max()[0]
                ax.set_ylim(bottom=-velocity, top=velocity)
        if return_df:
            return hankel_full
    
    def _forecast_prep(self, singular_values=None):
        self.X_com_hat = np.zeros(self.complete_dimensions)
        self.verticality_coefficient = 0
        self.forecast_orthonormal_base = {}
        if singular_values:
            try:
                for i in singular_values:
                    self.forecast_orthonormal_base[i] = self.orthonormal_base[i]
            except:
                if singular_values==0:
                    self.forecast_orthonormal_base[0] = self.orthonormal_base[0]
                else:
                    raise('Please pass in a list/array of singular value indices to use for forecast')
        else:
            self.forecast_orthonormal_base = self.orthonormal_base
        self.R = np.zeros(self.forecast_orthonormal_base[0].shape)[:-1]
        for Pi in self.forecast_orthonormal_base.values():
            self.X_com_hat += Pi*Pi.T*self.X_com
            pi = np.ravel(Pi)[-1]
            self.verticality_coefficient += pi**2
            self.R += pi*Pi[:-1]
        self.R = m(self.R/(1-self.verticality_coefficient))
        self.X_com_tilde = self.diagonal_averaging(self.X_com_hat)
        
    def forecast_recurrent(self, steps_ahead=12, singular_values=None, plot=False, return_df=False, **plotargs):
        '''Forecast from last point of original time series up to steps_ahead using recurrent methodology
        This method also fills any missing data from the original time series.'''
        try:
            self.X_com_hat
        except(AttributeError):
            self._forecast_prep(singular_values)
        self.ts_forecast = np.array(self.ts_v[0])
        for i in range(1, self.ts_N+steps_ahead):
            try:
                if np.isnan(self.ts_v[i]):
                    x = self.R.T*m(self.ts_forecast[max(0,i-self.R.shape[0]): i]).T
                    self.ts_forecast = np.append(self.ts_forecast,x[0])
                else:
                    self.ts_forecast = np.append(self.ts_forecast,self.ts_v[i])
            except(IndexError):
                x = self.R.T*m(self.ts_forecast[i-self.R.shape[0]: i]).T
                self.ts_forecast = np.append(self.ts_forecast, x[0])
        self.forecast_N = i+1
        new_index = pd.date_range(start=self.ts.index.min(),periods=self.forecast_N, freq=self.freq)
        forecast_df = pd.DataFrame(self.ts_forecast, columns=['Forecast'], index=new_index)
        forecast_df['Original'] = np.append(self.ts_v, [np.nan]*steps_ahead)
        if plot:
            forecast_df.plot(title='Forecasted vs. original time series', **plotargs)
        if return_df:
            return forecast_df
            

## NBEATS UTILS
# plot utils.
def plot_scatter(*args, **kwargs):
    plt.plot(*args, **kwargs)
    plt.scatter(*args, **kwargs)


# simple batcher.
def data_generator(x_full, y_full, bs):
    def split(arr, size):
        arrays = []
        while len(arr) > size:
            slice_ = arr[:size]
            arrays.append(slice_)
            arr = arr[size:]
        arrays.append(arr)
        return arrays

    while True:
        for rr in split((x_full, y_full), bs):
            yield rr

# trainer
def train_100_grad_steps(data, device, net, optimiser):
    global_step = load(net, optimiser)
    for x_train_batch, y_train_batch in data:
        global_step += 1
        optimiser.zero_grad()
        net.train()
        _, forecast = net(torch.tensor(x_train_batch, dtype=torch.float).to(device))
        loss = F.mse_loss(forecast, torch.tensor(y_train_batch, dtype=torch.float).to(device))
        loss.backward()
        optimiser.step()
        # if global_step % 30 == 0:
        #     print(f'grad_step = {str(global_step).zfill(6)}, tr_loss = {loss.item():.6f}, te_loss = ')
        if global_step > 0 and global_step % 100 == 0:
            with torch.no_grad():
                save(net, optimiser, global_step)
            break

# loader/saver for checkpoints.
def load(model, optimiser):
    if os.path.exists(CHECKPOINT_NAME):
        checkpoint = torch.load(CHECKPOINT_NAME)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        grad_step = checkpoint['grad_step']
        #print(f'Restored checkpoint from {CHECKPOINT_NAME}.')
        return grad_step
    return 0

def save(model, optimiser, grad_step):
    torch.save({
        'grad_step': grad_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimiser.state_dict(),
    }, CHECKPOINT_NAME)

# evaluate model on test data and produce some plots.
def eval_test(backcast_length, forecast_length, net, norm_constant, test_losses, x_test, y_test):
    net.eval()
    _, forecast = net(torch.tensor(x_test, dtype=torch.float))
    test_losses.append(F.mse_loss(forecast, torch.tensor(y_test, dtype=torch.float)).item())
    p = forecast.detach().numpy()
    subplots = [221, 222, 223, 224]
    plt.figure(1)
    for plot_id, i in enumerate(np.random.choice(range(len(x_test)), size=4, replace=False)):
        ff, xx, yy = p[i] * norm_constant, x_test[i] * norm_constant, y_test[i] * norm_constant
        plt.subplot(subplots[plot_id])
        plt.grid()
        plot_scatter(range(0, backcast_length), xx, color='b')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), yy, color='g')
        plot_scatter(range(backcast_length, backcast_length + forecast_length), ff, color='r')
    plt.show()
    