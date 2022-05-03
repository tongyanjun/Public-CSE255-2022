import numpy as np
from pylab import *
from scipy.stats import norm, uniform
import scipy



def error_curve(r_pos,r_neg,verticals=[],_xlim=None,_ylim=None):
    """ compute and plot the classification error curve for all rules of the form l>T
        r_pos,r_neg - arrays containing the positive ad the negative examples
        verticals - a list of pairs defining vertical lines (name,loc) loc is x location and name is a string
    """
    assert r_pos.shape[0]==r_neg.shape[0], 'r_pos and r_neg must have the same size'
    neg_data=np.stack([r_neg,ones(r_neg.shape)]).T
    pos_data=np.stack([r_pos,-ones(r_pos.shape)]).T
    data=np.concatenate([pos_data,neg_data],axis=0)

    sorted_data=data[data[:,0].argsort(),:]
    n=r_neg.shape[0]
    cs=cumsum(sorted_data[:,1])
    cs=n-cs
    min_loc=sorted_data[argmin(cs),0]
    plot([min_loc,min_loc],[0,n],label='min train error')
    for name,loc in verticals:
        plot([loc,loc],[0,n],label=name)
    plot(sorted_data[:,0],cs,label='error curve')
    xlabel('x')
    ylabel('number of mistakes')
    if not _xlim is None:
        xlim(_xlim)
    if not _ylim is None:
        ylim(_ylim)
    grid()
    legend()

class Mixture:
    """Defines pdf and sampling over mixtures of distributions"""
    def __init__(self,List):
        s=0.0
        for p,D in List:
            assert type(p) == float
            assert type(D) == scipy.stats._distn_infrastructure.rv_frozen
            s+=p
        assert abs(s-1)<10e-5
        List=sorted(List,key=lambda X:X[0])
        self.List=List
    def pdf(self,x):
        answer=np.zeros(x.shape)
        for p,D in self.List:
            answer+= p*D.pdf(x)
        return(answer)
    def rvs(self,size):
        left=size
        Samples=[]
        for p,D in self.List:
            ns = int(round(p*size))
            if(ns >left):
                ns=left
            if(ns>0):
                sample=D.rvs(size=ns)
                Samples.append(sample)
                left-=ns
        return np.concatenate(Samples)
        
def TwoNormals(_dist_from_0=1, _nsigma=3):
    x = np.linspace(-_dist_from_0-_nsigma,_dist_from_0+_nsigma,100)
    negative=norm(loc=-_dist_from_0)
    positive=norm(loc=_dist_from_0)
    plot(x, positive.pdf(x), 'b-', lw=2, label='positive')
    plot(x, negative.pdf(x), 'r-', lw=2, label='negative')
    xlabel('x')
    ylabel('density')
    grid()
    legend();
    return negative,positive
