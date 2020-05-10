
import gpdpwmFit as gp


def empericalplt(obj, labels =True):
    """


    """
    extend = 1.5
    u = obj['threshold']
    data = gp.np.array(obj['exceedances'])
    sorted_ex = sorted(data)
    Shape = obj['Shape']
    Scale = obj['Scale']
    if (len(sorted_ex)==1):
        m=sorted_ex
    else:
        m=len(sorted_ex)
    a=3/8
    ypoints=(gp.np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    U = max(sorted_ex)*extend
    z = gp.qgpd(gp.np.linspace(0, 1, num = 1000), Shape, u, Scale)
    z= [gp.np.minimum(i, U) for i in z]
    z= [gp.np.maximum(i,u) for i in z]
    y = gp.pgpd(z, Shape, u, Scale) 

    plotdf=gp.pd.DataFrame({"Fu(x-u)":sorted_ex, "x[log Scale]": ypoints})
    # Labels:
    if labels:
        
        xaxis = "Fu(x-u)"
        yaxis = "x[log Scale]"
        title = "Excess Distribution"
        gp.sns.lineplot(x=xaxis, y=yaxis, data=plotdf).set_title(title)
        plt.show()
    else:
        xaxis = yaxis = title = ""
    # Plot:
        gp.sns.lineplot(x="Fu(x-u)", y="x[log Scale]", data=plotdf).set_title(title)
        plt.show()


def disttail (obj,labels = True): 
    """
    Description:
       Tail of Underlying Distribution
    
    Arguments:
        x - an object of class fGPDFIT
        labels - a logical flag. Should labels be printed?
    
    """
    # Settings:
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
    ypoints=(gp.np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    ypoints = (1 - prob) * (1 - ypoints)
    z = self.obj.qgpd(gp.np.linspace(0, 1, num = 1000), shape, u, scale)
    z= [gp.np.minimum(i, U) for i in z]
    z= [gp.np.maximum(i,u) for i in z]
    y = gp.np.array(self.obj.pgpd(z, shape, u, scale))
    y = (1 - prob) * (1 - y)
    ylist=[]
    for i in y:
        if i>0:
            ylist.append(i)

    plt.plot(sorted_ex, ypoints) 
    plt.show()

def tailestimategpd(obj,doplot=True):
"""
Description:
    Plots tail estimate from GPD model

Arguments:
    object - an object of class 'gpdpwmfit'

Example:
    object = gpdFit(as.timeSeries(data(danishClaims)), u = 10)
    gpdTailPlot(object)
"""
# Settings:
    extend=1.5
    x=obj['data']
    threshold = obj['threshold']
    data = obj['exceedances']
    shape = obj['Shape']
    scale = obj['Scale']
    sorted_ex= sorted(data)
    # Points:
    u=  threshold
    U=  max(data) * gp.np.maximum(1, extend)
    z = self.obj.qgpd(gp.np.linspace(0, 1, num = 501), mu = threshold, beta =scale ,xi = shape)
    z= gp.np.minimum(z, U)
    z= gp.np.maximum(z,u)

    invProb = gp.np.array(1 - len(data)/len(x))
    if (len(sorted_ex)==1):
        m=sorted_ex
    else:
        m=len(sorted_ex)
        a=1/2
        ypoints=(gp.np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    ypoints = invProb*(1-ypoints)
    y =[invProb.tolist()* j for j in [1- i for i in gp.np.array(self.obj.pgpd(z,  threshold,shape, scale)).flatten()]]
    # Parameters:
    shape = shape
    scale = scale * invProb**shape
    location = threshold - (scale*(invProb**(-shape)-1))/shape
    

    # View Plot:
    if doplot:


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

        plt.ylim = plt.yticks(gp.np.arange(0,1.5,0.1))
        plt.plot(zz, yy)
        plt.show()
    
def residualplot(self,x,labels=True)
"""
Description:
    Quantile-Quantile Plot of GPD Residuals

Arguments:
    x - an object of class fGPDFIT
    labels - a logical flag. Should labels be printed?


"""
#Data:
data = obj['residuals']
sorted_data = sorted(data)

# Labels:
if labels: 
    if (len(data)==1):
        m=data
    else:
        m=len(data)
        a=3/8
        ypoints=(gp.np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    y = si.expon.ppf(ypoints)
    plt.plot(x = sorted, y = y)
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
        ypoints=(gp.np.linspace(start=1,stop=m, num=m) - a)/(m + (1-a)-a)
    y = si.expon.ppf(ypoints)
    plt.plot(x = sorted, y = y)
    plt.show()

ob=gpdpwmplots(gp.A)
ob.tail(doplot=True)

