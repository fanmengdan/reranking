import json
from xml.dom import minidom

print '\n===================== DATA INTEGRITY TEST : STARTING... =====================\n'

config = json.load(open('config.json', 'r'))
dataPath = config['TRAIN_NN']['dataPath']
fileList = config['TRAIN_NN']['fileList']

for xmlFile in fileList:
    doc = minidom.parse(dataPath + xmlFile)
    x = doc.getElementsByTagName("Thread")
    y = doc.getElementsByTagName("RelQuestion")
    print xmlFile, (len(x), len(y))
    z = zip(x, y)
    for i in range( len(z) ):
        if z[i][0].getAttribute("THREAD_SEQUENCE") != z[i][1].getAttribute("RELQ_ID"):
            print 'DATA CORRUPTED : @LINE', i + 1
            print '\n===================== DATA INTEGRITY TEST : ROYAL FAIL  =====================\n'
            exit()

print '\n===================== DATA INTEGRITY TEST : SUCCESSFULL =====================\n'