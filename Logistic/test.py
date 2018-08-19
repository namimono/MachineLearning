import log
import numpy as ny
dataset,labels=log.loadDataSet('data')
weight=log.gradAscent(dataset, labels)
weight=log.stocGradAscent(ny.array(dataset), labels)
log.plotBestFit(weight, dataset, labels)
