import numpy as np
import sys
import re
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker
import operator
from collections import defaultdict

def do_plot(fname, title, lab, y, x, c, alpha, is_log):
    plt.rcParams["figure.figsize"] = (5.5,4.8)
    plt.figure()
    if is_log:
      plt.loglog()
    plt.plot(x,
             y,
             'ro')

    plt.title(title)
    plt.plot( np.arange(0.1, max(x)*2, 0.1),
                   func(np.arange(0.1, max(x)*2, 0.1),c,alpha),
                  label=lab, color='#2266ff', linewidth=2)
    plt.legend()
    plt.xlabel("rank")
    plt.ylabel("frequency")
    if is_log:
      plt.xticks([1,10,100,1000])
      plt.xlim(left=0.8) 
      plt.ylim(top=0.3)
      plt.ylim(bottom=0.0001)
      plt.savefig('plot/plt_'+fname+'_log.png')
    else:
      topy=(max(y)*1.3)
      if topy<0.1:
        plt.yticks(np.arange(0, topy, 0.01))
      else:
        plt.yticks(np.arange(0, topy, 0.05))
      plt.xlim(left=-1*int(max(x)/50)) 
      plt.xlim(right=int(max(x)*1.3))
      plt.ylim(top=topy)
      plt.ylim(bottom=-0.002) 
      plt.savefig('plot/plt_'+fname+'_pow.png')

def print_freq_words(words_dict, fname):
  sorted_d = sorted(words_dict.items(), key=operator.itemgetter(1),reverse=True)
  res='FREQ '+fname+' '
  for i in range(0,5):
    res=res+' '+str(sorted_d[i][0])+':'+"{:.2f}".format(100*sorted_d[i][1])+'%'
  print res
  
def generate_csv(words_dict,fname):
  sorted_d = sorted(words_dict.items(), key=operator.itemgetter(1),reverse=True)
  f= open("csv/"+fname+".csv","w+")
  i=1
  for word,freq in sorted_d:
    f.write("%s,%d,%.7f\n" % (word,i,freq))
    i=i+1
  f.close() 

# power function
def func(myx, c, a):
  return c/np.power(myx,a)


if __name__ == '__main__':

    words = list()
    
    fname=re.sub(r'.*/','',sys.argv[1])
    with open(sys.argv[1], 'r') as myfile:
      line=myfile.readline()
      while (len(line)>0):
        tokens=line.split()
        for t in tokens:
          if t!='':
            words.append(t)
        line=myfile.readline()

    y = defaultdict(int)
    for i in words:
        y[i]+=1.0/float(len(words)) #percentage
    print_freq_words(y,fname)
    generate_csv(y,fname)

    # sort by decreasing frequency
    ydata = np.array(sorted(y.values(),reverse=True))
    xdata = np.array(xrange(1,len(y)+1))

    
    popt, pcov = scipy.optimize.curve_fit(func, xdata, ydata)
    print "popt:"+str(popt)
    print pcov

    c, alpha = popt
    print "c "+str(c)
    print 'alpha %.2f ' % alpha
    
    actual=list()
    expected=list()
    rank=1
    for ycount in ydata:
      actual.append(ycount)
      expected.append(func(rank,c,alpha))
      rank += 1
    for i in actual[:10]:
      sys.stdout.write("{:.5f} ".format(i))
    print
    for i in expected[:10]:
      sys.stdout.write("{:.5f} ".format(i))
    print

    
    MSE = np.square(np.subtract(actual,expected)).mean() 
    print 'MSE %.10f' % MSE
    
    RMSD = np.sqrt(MSE)
    print 'RMSD %.10f' % RMSD
    
    print 'range: ' +str(max(ydata)-min(ydata))
    
    NRMSD=RMSD/(max(ydata)-min(ydata))
    print fname+' NRMSD:%.3f   C:%.3f  alpha:%.3f' % (NRMSD,c,alpha)
    
    
    meanAbsError=np.mean(np.abs(np.array(actual) - np.array(expected) ))
    print 'meanAbsError %.7f' % meanAbsError
    
    meanAbsPercError=np.mean(100.0*np.abs(np.array(actual) - np.array(expected) )/np.array(expected))
    print 'meanAbsPercError %.7f' % meanAbsPercError
    
    lab='%.3f/x^%.3f' % (c,alpha)
    title=fname + "   NRMSD:%.3f" % NRMSD
    # log log plot
    do_plot(fname, title, lab, ydata, xdata, c, alpha, True)
    # linear scale plot
    #do_plot(fname, title, lab, ydata, xdata, c, alpha, False)
