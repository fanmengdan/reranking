from xml.dom import minidom

print '\n===================== DATA INTEGRITY TEST : STARTING... =====================\n'

fileList = [ "SemEval2016-Task3-CQA-QL-train-part1-subtaskA.xml", "SemEval2016-Task3-CQA-QL-train-part2-subtaskA.xml", "SemEval2016-Task3-CQA-QL-dev-subtaskA.xml", \
"SemEval2015-Task3-CQA-QL-train-reformatted-excluding-2016-questions-cleansed.xml", "SemEval2015-Task3-CQA-QL-dev-reformatted-excluding-2016-questions-cleansed.xml", \
"SemEval2015-Task3-CQA-QL-test-reformatted-excluding-2016-questions-cleansed.xml" ]

datapath = '/media/sandeshc/Windows_OS/Sandesh/MTP/Data/semeval2016-task3/useful/train-NN/'

for xmlFile in fileList:
    doc = minidom.parse(datapath + xmlFile)
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