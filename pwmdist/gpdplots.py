
from pwmdist.gpdpwmFit import *



def empericalplt(obj, labels =True):
    """
    1) Description:
        Empirical Distribution Plot
    2) Input parameters:
        obj = Fitbygpdpwm function output 
        labels = By default True
    3) Example:
        obj= Fitbygpdpwm(data, ci, threshold)
        empericalplt(obj)

    """
    extend = 1.5
    u = obj['threshold']
    data = np.array(obj['exceedances'])
    sorted_ex = sorted(data)
    Shape = obj['Shape']
    Scale = obj['Scale']
    if (len(sorted_ex)==1):
        m=sorted_ex
    else:
        m=len(sorted_ex)
    a=3/8
    ypoints=(np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    U = max(sorted_ex)*extend
    z = qgpd(np.linspace(0, 1, num = 1000), Shape, u, Scale)
    z= [np.minimum(i, U) for i in z]
    z= [np.maximum(i,u) for i in z]
    y = pgpd(z, Shape, u, Scale) 

    plotdf=pd.DataFrame({"Fu(x-u)":sorted_ex, "x[log Scale]": ypoints})
    # Labels:
    if labels:
        
        xaxis = "Fu(x-u)"
        yaxis = "x[log Scale]"
        title = "Excess Distribution"
        sns.lineplot(x=xaxis, y=yaxis, data=plotdf).set_title(title)
        plt.show()
    else:
        xaxis = yaxis = title = ""
    # Plot:
        sns.lineplot(x="Fu(x-u)", y="x[log Scale]", data=plotdf).set_title(title)
    return plt.show()


def disttail(obj,labels = True): 
    """
    1) Description:
       Tail of Underlying Distribution
    
    2) Arguments:
        x - an object of class fGPDFIT
        labels - a logical flag. Should labels be printed?

    3) Example:
        obj= Fitbygpdpwm(data, ci, threshold)
        disttail(obj)

    """
    extend = 1.5
    u = obj['threshold']
    data=obj['data']
    sorted_ex = sorted(obj['exceedances'])
    prob = obj['prob']
    shape = xi = obj['Shape']
    scale = obj['Scale']* (1-prob)**shape
    location = u - (scale*((1 - prob)**(-shape)-1))/shape

    # Labels:
    if labels:
        xlab = "x [log scale]"
        ylab = "1-F(x) [log scale]"
        main = "Tail of Underlying Distribution"
    else:
        xlab = ylab = main = ""

    # Plot:
    U = max(sorted_ex)*extend
    if (len(sorted_ex)==1):
        m=sorted_ex
    else:
        m=len(sorted_ex)
    a=3/8
    ypoints=(np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    ypoints = (1 - prob) * (1 - ypoints)
    z = qgpd(np.linspace(0, 1, num = 1000), shape, u, scale)
    z= [np.minimum(i, U) for i in z]
    z= [np.maximum(i,u) for i in z]
    y = np.array(pgpd(z, shape, u, scale))
    y = (1 - prob) * (1 - y)
    ylist=[]
    for i in y:
        if i>0:
            ylist.append(i)

    plt.plot(sorted_ex, ypoints) 
    plt.title("Tail of Underlying Distribution")
    return plt.show()

def tailestimategpd(obj,labels=True):
    """
    1) Description:
        Plots tail estimate from GPD model

    2) Arguments:
        obj = Fitbygpdpwm function output 
        labels = By default True
    3) Example:
        obj= Fitbygpdpwm(data, ci, threshold)
        tailestimategpd(obj)
    """

    extend=1.5
    x=obj['data']
    threshold = obj['threshold']
    data = obj['exceedances']
    shape = obj['Shape']
    scale = obj['Scale']
    sorted_ex= sorted(data)
    # Points:
    u=  threshold
    U=  max(data) * np.maximum(1, extend)
    z = qgpd(np.linspace(0, 1, num = 501), location = threshold, scale =scale ,shape = shape)
    z= np.minimum(z, U)
    z= np.maximum(z,u)

    invProb = np.array(1 - len(data)/len(x))
    if (len(sorted_ex)==1):
        m=sorted_ex
    else:
        m=len(sorted_ex)
        a=1/2
        ypoints=(np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    ypoints = invProb*(1-ypoints)
    y =[invProb.tolist()* j for j in [1- i for i in np.array(pgpd(z,  threshold,shape, scale)).flatten()]]
    # Parameters:
    shape = shape
    scale = scale * invProb**shape
    location = threshold - (scale*(invProb**(-shape)-1))/shape
    

    # View Plot:
    if labels:


        plt.plot(sorted_ex, ypoints,'o')
        zz=[]
        for idx,i in enumerate(y):
            if i>=0:
                zz.append(z[idx])
        else:
            pass
        yy=[]
        for i in y:
            if i>=0:
                yy.append(i)

        plt.ylim = plt.yticks(np.arange(0,1.5,0.1))
        plt.plot(zz, yy)
        plt.title("Tail estimate from GPD model")
    return plt.show()
    
def residualplot(obj,labels=True):
    """
    1)Description:
        Quantile-Quantile Plot of GPD Residuals

    2) Arguments:
        obj = Fitbygpdpwm function output
        labels - a logical flag. checks if labels should be printed.

    3) Example:
        obj = Fitbygpdpwm(data, ci, threshold)
        residualplot(obj)
    """
    data = obj['residuals']
    sorted_data = sorted(data)

    # Labels:
    if labels: 
        if (len(data)==1):
            m=data
        else:
            m=len(data)
            a=3/8
            ypoints=(np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
        y = si.stats.expon.ppf(ypoints)
        plt.scatter(sorted_data,  y)
        A = np.vstack([sorted_data, np.ones(len(sorted_data))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        plt.plot(sorted_data, m*np.array(sorted_data) + c, 'r', label='Fitted line')
        plt.xlabel("Ordered Data")
        plt.ylabel("Exponential Quantiles")
        plt.title("QQ-Plot of Residuals")
        plt.show()
    else:
        if (len(data)==1):
            m=data
        else:
            m=len(data)
            a=3/8
            ypoints=(np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
        y = si.stats.expon.ppf(ypoints)
        plt.plot(sorted, y)
    return plt.show()
