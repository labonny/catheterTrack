#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Read and plot catheter raw data from RthReconImageExporter

    Reads a single raw data file exported from RTHawk. This is assumed to be a catheter
    projection file. The first projection is plotted and subsequent projections can
    be selected for plotting.

    Plots in png format and/or coordinates & physiology data in text format can be output instead using
    the "-p" and "-c" options.

    Note that respiratory data from RTHawk is scaled by 10^5 to give an integer (the respiratory information
    from RTHawk is a float between 0 and 1).

    Usage examples:
        ./readRthRaw.py data/file-0000.projections

        ./readRthRaw.py data/file-0000.projections -p

        ./readRthRaw.py data/file-0000.projections -c
        
    Note: to use the above commands in the terminal, 'python' must proceed the commands

"""

import sys, struct
import math
import scipy
import scipy.fftpack
import pylab
from matplotlib.widgets import Button
from optparse import OptionParser
import os
import snrCalc

float_bytes = 8 #These are being written on a 64-bit system

def readHeader(fp):
  hdr = fp.read(44) # 3 32-bit ints + 1 64-bit int + 3 64-bit floats = 44 bytes
  if not hdr:
    #print("reached EOF")
    return (0,0,0,0,0,0,0)
  else:
    xsize = struct.unpack('>i',hdr[0:4])[0]
    ysize = struct.unpack('>i',hdr[4:8])[0]
    zsize = struct.unpack('>i',hdr[8:12])[0]
    fov = struct.unpack('>d',hdr[12:20])[0]
    timestamp = struct.unpack('>q',hdr[20:28])[0]
    trig = struct.unpack('>d',hdr[28:36])[0]
    resp = struct.unpack('>d',hdr[36:44])[0]
    return (xsize,ysize,zsize,fov,trig,resp,timestamp)

def readLegacy2Header(fp):
  hdr = fp.read(36) # 3 32-bit ints + 3 64-bit floats = 36 bytes
  if not hdr:
    #print("reached EOF")
    return (0,0,0,0,0,0)
  else:
    xsize = struct.unpack('>i',hdr[0:4])[0]
    ysize = struct.unpack('>i',hdr[4:8])[0]
    zsize = struct.unpack('>i',hdr[8:12])[0]
    fov = struct.unpack('>d',hdr[12:20])[0]
    trig = struct.unpack('>d',hdr[20:28])[0]
    resp = struct.unpack('>d',hdr[28:36])[0]
    return (xsize,ysize,zsize,fov,trig,resp)
    
def readLegacyHeader(fp):
  hdr = fp.read(20) # 3 32-bit ints + 1 64-bit floats = 20 bytes
  if not hdr:
    #print("reached EOF")
    return (0,0,0,0)
  else:
    xsize = struct.unpack('>i',hdr[0:4])[0]
    ysize = struct.unpack('>i',hdr[4:8])[0]
    zsize = struct.unpack('>i',hdr[8:12])[0]
    fov = struct.unpack('>d',hdr[12:20])[0]
    return (xsize,ysize,zsize,fov)

class ProjectionPlot:
    def __init__(self,fts,xsize,fov,mode='magnitude',tickDistance=100,trigTimes=[],respArr=[], timestamps=[]):
        self.fts = fts
        self.xsize = xsize
        self.fov = fov
        self.mode = mode
        self.tickDistance = tickDistance
        self.makeTicks()
        self.index = 0
        self.plots = []
        self.axis = {0:'X',1:'Y',2:'Z'}
        self.stemMarkers = []
        self.stemBase = []
        self.stemLines = []
        self.trigTimes = trigTimes
        self.useTrig = len(trigTimes) > 0
        self.respArr = respArr
        self.useResp = len(respArr) > 0
        self.timeStamps = timestamps
        self.useTimeStamps = len(timestamps) > 0
        
    def showProj(self,frame, savePlots=False, saveCoords=False, coordFile=None, separateSNR=False):
      # fts: all the fourier-transformed projections in one array; x, y, and z each are in their own row
      # frame: which projection to show
      self.index = frame*3
      useTrig = False
      if self.useTrig:
        trig = self.trigTimes[frame]
      if self.useResp:
        resp = self.respArr[frame]
      if self.useTimeStamps:
        print('0: 0')

      del self.plots[:]
      self.clearStems()
      #print "Index " + str(index)
      if len(self.fts) < self.index+3 or self.index < 0:
        print("Frame " + str(frame) + " does not exist")
        return
      coords = []
      snrs = []
      if savePlots:
        pylab.figure(figsize=(13,6))

      for i in range(0,3):
        axes=pylab.subplot('13'+str(1+i))
        pylab.subplots_adjust(bottom=0.2)
        if (self.mode == "phase"):
          self.plots[i].append( pylab.plot(scipy.angle(self.fts[self.index+i])) )
          pylab.title(self.axis[i] + ' Phase Projection');pylab.xticks(self.tick_locs,self.tick_labels)
        else:
          mag = abs(self.fts[self.index+i])
          peak = max(mag)
          peakInd = list(mag).index(peak)
          self.plots.append( pylab.plot(mag) )
          snr = snrCalc.getSNR(mag,peak)
          snrs.append(snr) 
          pylab.title(self.axis[i] + ' Magnitude Projection');
          pylab.ylim([0,500]);  ## changed from 100 --> 500
          axes.set_autoscaley_on(False);
          pylab.xticks(self.tick_locs,self.tick_labels); 
          stem_marker, stem_lines, stem_base = pylab.stem([peakInd],[peak],'r-','ro');
          self.stemMarkers.append(stem_marker)
          self.stemBase.append(stem_base)
          self.stemLines.append(stem_lines)
          xres = self.fov/self.xsize
          coords.append(xres*(peakInd-len(mag)/2))
          pylab.xlabel(self.axis[i]+':'+'{0:.3}'.format(coords[i])+' mm')
      if savePlots:
        pylab.savefig('proj{0:04d}.png'.format(frame))
        self.clearStems()
        pylab.clf()
        pylab.close()
      if saveCoords and not coordFile is None:
        coordFile.write("%0.1f %0.1f %0.1f" % (coords[0], coords[1], coords[2]))
        if separateSNR:
          coordFile.write(" %0.1f %0.1f %0.1f" % (snrs[0], snrs[1], snrs[2]))
        else:
          coordFile.write(" %d" % (min(snrs)))
        if self.useTimeStamps:
          coordFile.write(" %d" % (timestamp))
        if self.useTrig:
          coordFile.write(" %d" % (trig))
        if self.useResp:
          coordFile.write(" %d" % (resp * (10**5)))
        coordFile.write("\n")
      elif not savePlots and not saveCoords:
        pylab.draw()
        
    def makeTicks(self):
        zeroPixel = (self.xsize + 1)/2.0
        tickIncr = self.tickDistance/(self.fov/self.xsize)
        numTicks = int(math.floor(self.xsize / tickIncr))
        self.tick_locs = []
        self.tick_labels = []
        currLabel = -1*self.fov/2.0
        currLoc = 0
        for i in range(0,numTicks):
            self.tick_labels.append(str(currLabel))
            currLabel += self.tickDistance
            self.tick_locs.append(currLoc)
            currLoc += tickIncr

    def clearStems(self):
        for marker in self.stemMarkers:
            marker.remove()
        for base in self.stemBase:
            base.remove()
        for line in self.stemLines:
            line[0].remove()

        del self.stemMarkers[:]
        del self.stemBase[:]
        del self.stemLines[:]

    def redraw(self):
        self.clearStems()        

        for i in range(0,3):
            axes = pylab.subplot('13'+str(1+i))
            if (self.mode == "phase"):
                self.plots[i][0].set_ydata(scipy.angle(self.fts[self.index+i]))
            else:
                mag = abs(self.fts[self.index+i])
                self.plots[i][0].set_ydata(mag)
                peak = max(mag)
                peakInd = list(mag).index(peak)
                stem_marker, stem_lines, stem_base = pylab.stem([peakInd],[peak],'r-','ro');
                self.stemMarkers.append(stem_marker)
                self.stemBase.append(stem_base)
                self.stemLines.append(stem_lines)
                xres = self.fov/self.xsize
                pylab.xlabel(self.axis[i]+':'+'{0:.3}'.format(xres*(peakInd-len(mag)/2))+' mm')

            pylab.draw()

    def next(self,event):
      self.index += 3
      self.index = self.index % len(self.fts)
      if self.useTimeStamps:
        idx = int(round(math.floor(self.index/3)))
        #print(str(idx)+': '+str(self.timeStamps[idx]-self.timeStamps[0]))
      self.redraw()

    def prev(self,event):
      self.index -= 3
      self.index = self.index % len(self.fts)
      if self.useTimeStamps:
        idx = int(round(math.floor(self.index/3)))
        #print(str(idx)+': '+str(self.timeStamps[idx]-self.timeStamps[0]))
      self.redraw()

# Returns: (coordinate of peak, peak amplitude, SNR)
# Given: a single fourier-transformed projection, the FOV, and the size of the projection
def getPeakAndSNR(projectionFT, fov, xsize):
    mag = abs(projectionFT)
    peak = max(mag)
    peakInd = list(mag).index(peak)
    snr = snrCalc.getSNR(mag,peak)
    xres = fov/xsize
    coord = xres*(peakInd-len(mag)/2)
    return coord,peak,snr
    
def readProjections(fname,legacy=False,legacy2=False,show=True):
    fp = open(fname,"rb")
    projections = [] # array of tuples, where each tuple is a series of complex floats
    projComplex = []
    triggerTimes = [] #array of trigger times, one triggerTime per each triplet of projections
    respPhases = []
    timestamps = []
    projNum = 0
    projSize = 0
    first = True
    xsize = ysize = zsize = fieldOfView = 0;
    done = False
    while not done:
        xs = ys = zs = fov = 0
        if legacy:
            xs,ys,zs,fov=readLegacyHeader(fp)
        elif legacy2:
            xs,ys,zs,fov,trig,resp=readLegacy2Header(fp)
            triggerTimes.append(trig)
            respPhases.append(resp)
        else:
            xs,ys,zs,fov,trig,resp,timestamp=readHeader(fp)
            triggerTimes.append(trig)
            respPhases.append(resp)
            timestamps.append(timestamp)
        if (xs == 0 or ys == 0 or zs == 0):
            done = True;
            break
        if first:
            xsize = xs
            ysize = ys
            zsize = zs
            fieldOfView = fov
            projSize = xs*ys*zs*2
            projByteSize = projSize*float_bytes
            proj = fp.read(projByteSize)
        if proj is None or len(proj) < projByteSize:
            #print("Could not read projection " + str(projNum) + " stopping here.")
            break
        projections.append( struct.unpack('>'+str(projSize)+'d',proj[0:projByteSize]) )
        projNum+=1
    #if show==True:
        #print("Read " + str(projNum) + " projections...",)
        #print("x size = " + str(xsize) + ", y size = " + str(ysize) + ", z size = " + str(zsize) + ", fov = " + str(fieldOfView))
    # NOTE: each projection in projComplex and projections contains the x, y and z projections
    for proj in range(0,projNum):
        projComplex.append([])
        for i in range(0,projSize,2):
            projComplex[proj].append( complex(projections[proj][i],projections[proj][i+1]) )
    return xsize,ysize,zsize,fieldOfView,projNum,triggerTimes,respPhases,timestamps,projComplex

def reconstructProjections(projComplex,xsize,ysize,zf=0):
    fts = []
    for projection in projComplex:
        # split into 'ysize' (3) projections
        for y in range(1,ysize+1):
            axis = projection[xsize*(y-1):xsize*y]
            if xsize<zf:
                while len(axis)<zf:
                    axis.append(complex(0,0))
            inverseft = scipy.fftpack.ifft(scipy.fftpack.ifftshift(axis)) #,npts)
            fts.append( scipy.fftpack.fftshift(inverseft) )
    return fts

def main():
    parser = OptionParser(usage=__doc__)
    parser.add_option("-p", "--plot-save", action="store_true", dest="saveplots",help="save plots to files, no gui", default=False)
    parser.add_option("-c", "--coord-save", action="store_true", dest="savecoords",help="save coordinates to files, no gui", default=False)
    parser.add_option("-l", "--legacy", action="store_true", dest="legacy",help="read legacy files with no trig, resp, or timestamp values", default=False)
    parser.add_option("-m", "--legacy2", action="store_true", dest="legacy2",help="read legacy files with trig and resp but no timestamp values", default=False)
    parser.add_option("-s", "--separate-snrs", action="store_true", dest="separateSNR",help="output snrs for each axis to coordinate files (only valid for coordinate saving mode)", default=False )
    (options,args) = parser.parse_args()

    if (len(args) < 1):
        print(parser.print_help())
        sys.exit(0)
    rawFile = args[0]
    dbase = os.path.splitext(rawFile)[0] + '.sqlite3'

    xsize,ysize,zsize,fieldOfView,projNum,triggerTimes,respPhases,timestamps,projComplex = readProjections(rawFile,options.legacy,options.legacy2)

    #print("Num projections " + str(len(projComplex)))
    fts = reconstructProjections(projComplex,xsize,ysize)
    #print("Num ffts " + str(len(fts)))

    if len(fts) > 0:
      plotter = ProjectionPlot(fts,xsize,fieldOfView,trigTimes=triggerTimes,respArr=respPhases,timestamps=timestamps)

    #Save to files:
    if (options.saveplots or options.savecoords) and len(fts) > 0:
      coordFile = None
      if options.savecoords:
        fbase,fext = os.path.splitext(rawFile)
        coordFile = open(fbase + '-coords.txt','w')
      for i in range(len(projComplex)):
        plotter.showProj(i,options.saveplots,options.savecoords,coordFile,options.separateSNR)
        sys.stdout.write("\rSaved projection %i" % i)
        sys.stdout.flush()
      print("\nDone.")
    elif len(fts) > 0:
      plotter.showProj(0)
      axprev = pylab.axes([0.7, 0.02, 0.1, 0.075])
      axnext = pylab.axes([0.81, 0.02, 0.1, 0.075])
      bnext = Button(axnext, 'Next')
      bnext.on_clicked(plotter.next)
      bprev = Button(axprev, 'Previous')
      bprev.on_clicked(plotter.prev)

      pylab.show()
    
    sys.exit(0)
    
if __name__ == "__main__":
    main()