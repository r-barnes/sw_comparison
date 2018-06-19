/**
 * UGENE - Integrated Bioinformatics Tools.
 * Copyright (C) 2008-2018 UniPro <ugene@unipro.ru>
 * http://ugene.net
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301, USA.
 */

#include <U2Core/AppContext.h>
#include <U2Core/Counter.h>
#include <U2Core/DNASequenceUtils.h>
#include <U2Core/FileAndDirectoryUtils.h>
#include <U2Core/GUrlUtils.h>
#include <U2Core/IOAdapter.h>
#include <U2Core/IOAdapterUtils.h>
#include <U2Core/TaskSignalMapper.h>
#include <U2Core/U2SafePoints.h>

#include <U2Designer/DelegateEditors.h>

#include <U2Formats/BAMUtils.h>
#include <U2Formats/FastqFormat.h>

#include <U2Lang/ActorPrototypeRegistry.h>
#include <U2Lang/BaseActorCategories.h>
#include <U2Lang/BaseAttributes.h>
#include <U2Lang/BaseSlots.h>
#include <U2Lang/BaseTypes.h>
#include <U2Lang/IntegralBusModel.h>
#include <U2Lang/WorkflowEnv.h>
#include <U2Lang/WorkflowMonitor.h>

#include "FASTQWorkersLibrary.h"

#define MAX_PHRED33_VALUE 74

namespace U2 {
namespace LocalWorkflow {

///////////////////////////////////////////////////////////////
//CASAVAFilter
const QString CASAVAFilterWorkerFactory::ACTOR_ID("CASAVAFilter");

/************************************************************************/
/* CASAVAFilterPrompter */
/************************************************************************/
QString CASAVAFilterPrompter::composeRichDoc() {
    IntegralBusPort* input = qobject_cast<IntegralBusPort*>(target->getPort(BaseNGSWorker::INPUT_PORT));
    const Actor* producer = input->getProducer(BaseSlots::URL_SLOT().getId());
    QString unsetStr = "<font color='red'>"+tr("unset")+"</font>";
    QString producerName = tr("<u>%1</u>").arg(producer ? producer->getLabel() : unsetStr);

    QString doc = tr("Filters FASTQ reads from %1 generated by CASAVA 1.8.").arg(producerName);
    return doc;
}

/************************************************************************/
/* CASAVAFilterWorkerFactory */
/************************************************************************/
void CASAVAFilterWorkerFactory::init() {
    Descriptor desc( ACTOR_ID, CASAVAFilterWorker::tr("CASAVA FASTQ Filter"),
        CASAVAFilterWorker::tr("Reads in FASTQ file produced by CASAVA 1.8 contain 'N' or 'Y' as a part of an idetifier. 'Y' if a read if filtered, 'N' if the read if the read is not filtered. The workflow cleans up the filtered reads.\n"
                           "For example:\n"
                           "@HWI-ST880:181:D1WRUACXX:8:1102:4905:2125 1:N:0:TAAGGG\n"
                           "CTTACATAACTACTGACCATGCTCTCTCTTGTCTGTCTCTTATACACATCT\n"
                           "+\n"
                           "111442222322324232AAFFHIJJJJJJIHIIF111CGGFHIG???FGB\n"
                           "@HWI-ST880:181:D1WRUACXX:8:1102:7303:2101 1:Y:0:TAAGGG\n"
                           "TCCTTACTGTCTGAGCAATGGGATTCCATCTTTTACGATCTAGACATGGCT\n"
                           "+\n"
                           "11++4222322<CEE2<:3<3333333<A<CAFE:1?C??::C?00?BD90.\n") );

    QList<PortDescriptor*> p;
    {
        Descriptor inD(BaseNGSWorker::INPUT_PORT, CASAVAFilterWorker::tr("Input File"),
            CASAVAFilterWorker::tr("Set of FASTQ reads files"));
        Descriptor outD(BaseNGSWorker::OUTPUT_PORT, CASAVAFilterWorker::tr("Output File"),
            CASAVAFilterWorker::tr("Output FASTQ files"));

        QMap<Descriptor, DataTypePtr> inM;
        inM[BaseSlots::URL_SLOT()] = BaseTypes::STRING_TYPE();
        p << new PortDescriptor(inD, DataTypePtr(new MapDataType("cf.input-url", inM)), true);

        QMap<Descriptor, DataTypePtr> outM;
        outM[BaseSlots::URL_SLOT()] = BaseTypes::STRING_TYPE();
        p << new PortDescriptor(outD, DataTypePtr(new MapDataType("cf.output-url", outM)), false, true);
    }

    QList<Attribute*> a;
    {
        Descriptor outDir(BaseNGSWorker::OUT_MODE_ID, CASAVAFilterWorker::tr("Output folder"),
            CASAVAFilterWorker::tr("Select an output folder. <b>Custom</b> - specify the output folder in the 'Custom folder' parameter. "
            "<b>Workflow</b> - internal workflow folder. "
            "<b>Input file</b> - the folder of the input file."));

        Descriptor customDir(BaseNGSWorker::CUSTOM_DIR_ID, CASAVAFilterWorker::tr("Custom folder"),
            CASAVAFilterWorker::tr("Select the custom output folder."));

        Descriptor outName(BaseNGSWorker::OUT_NAME_ID, CASAVAFilterWorker::tr("Output file name"),
            CASAVAFilterWorker::tr("A name of an output file. If default of empty value is provided the output name is the name of the first file with additional extention."));

        a << new Attribute(outDir, BaseTypes::NUM_TYPE(), false, QVariant(FileAndDirectoryUtils::WORKFLOW_INTERNAL));
        Attribute* customDirAttr = new Attribute(customDir, BaseTypes::STRING_TYPE(), false, QVariant(""));
        customDirAttr->addRelation(new VisibilityRelation(BaseNGSWorker::OUT_MODE_ID, FileAndDirectoryUtils::CUSTOM));
        a << customDirAttr;
        a << new Attribute( outName, BaseTypes::STRING_TYPE(), false, QVariant(BaseNGSWorker::DEFAULT_NAME));
    }

    QMap<QString, PropertyDelegate*> delegates;
    {
        QVariantMap directoryMap;
        QString fileDir = CASAVAFilterWorker::tr("Input file");
        QString workflowDir = CASAVAFilterWorker::tr("Workflow");
        QString customD = CASAVAFilterWorker::tr("Custom");
        directoryMap[fileDir] = FileAndDirectoryUtils::FILE_DIRECTORY;
        directoryMap[workflowDir] = FileAndDirectoryUtils::WORKFLOW_INTERNAL;
        directoryMap[customD] = FileAndDirectoryUtils::CUSTOM;
        delegates[BaseNGSWorker::OUT_MODE_ID] = new ComboBoxDelegate(directoryMap);

        delegates[BaseNGSWorker::CUSTOM_DIR_ID] = new URLDelegate("", "", false, true);
    }

    ActorPrototype* proto = new IntegralBusActorPrototype(desc, p, a);
    proto->setEditor(new DelegateEditor(delegates));
    proto->setPrompter(new CASAVAFilterPrompter());

    WorkflowEnv::getProtoRegistry()->registerProto(BaseActorCategories::CATEGORY_NGS_BASIC(), proto);
    DomainFactory *localDomain = WorkflowEnv::getDomainRegistry()->getById(LocalDomainFactory::ID);
    localDomain->registerEntry(new CASAVAFilterWorkerFactory());
}

/************************************************************************/
/* CASAVAFilterWorker */
/************************************************************************/
CASAVAFilterWorker::CASAVAFilterWorker(Actor *a)
:BaseNGSWorker(a)
{

}

QVariantMap CASAVAFilterWorker::getCustomParameters() const{
    QVariantMap res;
    return res;
}

QString CASAVAFilterWorker::getDefaultFileName() const{
    return ".filtered.fastq";
}

Task *CASAVAFilterWorker::getTask(const BaseNGSSetting &settings) const {
    return new CASAVAFilterTask(settings);
}

//////////////////////////////////////////////////////
//CASAVAFilterTask
CASAVAFilterTask::CASAVAFilterTask(const BaseNGSSetting &settings)
    :BaseNGSTask(settings){
    GCOUNTER(cvar, tvar, "NGS:CASAVAFilterTask");
}

void CASAVAFilterTask::runStep(){
    int ncount = 0;
    int ycount = 0;

    QScopedPointer<IOAdapter> io  (IOAdapterUtils::open(settings.outDir + settings.outName, stateInfo, IOAdapterMode_Append));

    //1:N:0:TAAGGG
    QRegExp pattern (":Y:[^:]:");
    FASTQIterator iter(settings.inputUrl, stateInfo);
    if (stateInfo.hasError()) {
        return;
    }
    while(iter.hasNext()){
        if(stateInfo.isCoR()){
            return;
        }
        DNASequence seq = iter.next();
        QString comment = DNAInfo::getFastqComment(seq.info);
        if(pattern.indexIn(comment) != -1){
            ycount++;
        }else{
            FastqFormat::writeEntry(seq.getName() + " " + comment, seq, io.data(), "Writing error", stateInfo, false);
            ncount++;
        }
    }

    algoLog.info(QString("Discarded by CASAVA filter %1").arg(ycount));
    algoLog.info(QString("Accepted by CASAVA filter %1").arg(ncount));
    algoLog.info(QString("Total by CASAVA FILTER: %1").arg(ncount + ycount));
}

QStringList CASAVAFilterTask::getParameters(U2OpStatus &/*os*/) {
    QStringList res;
    return res;
}


///////////////////////////////////////////////////////////////
//QualityTrim
const QString FastqQualityTrimWorkerFactory::ACTOR_ID("QualityTrim");

static const QString QUALITY_ID("qual-id");
static const QString LEN_ID("len-id");
static const QString BOTH_ID("both-ends");

/************************************************************************/
/* QualityTrimPrompter */
/************************************************************************/
QString FastqQualityTrimPrompter::composeRichDoc() {
    IntegralBusPort* input = qobject_cast<IntegralBusPort*>(target->getPort(BaseNGSWorker::INPUT_PORT));
    const Actor* producer = input->getProducer(BaseSlots::URL_SLOT().getId());
    QString unsetStr = "<font color='red'>"+tr("unset")+"</font>";
    QString producerName = tr("<u>%1</u>").arg(producer ? producer->getLabel() : unsetStr);

    QString doc = tr("Trim input sequence %1 from the end, using the quality threshold.").arg(producerName);
    return doc;
}

/************************************************************************/
/* QualityTrimWorkerFactory */
/************************************************************************/
void FastqQualityTrimWorkerFactory::init() {
    Descriptor desc( ACTOR_ID, FastqQualityTrimWorker::tr("FASTQ Quality Trimmer"),
        FastqQualityTrimWorker::tr("The workflow scans each input sequence from the end to find the first position where the quality is greater or equal to the minimum quality threshold. "
                              "Then it trims the sequence to that position. If a the whole sequence has quality less than the threshold or the length of the output sequence less than "
                              "the minimum length threshold then the sequence is skipped.") );

    QList<PortDescriptor*> p;
    {
        Descriptor inD(BaseNGSWorker::INPUT_PORT, FastqQualityTrimWorker::tr("Input File"),
            FastqQualityTrimWorker::tr("Set of FASTQ reads files"));
        Descriptor outD(BaseNGSWorker::OUTPUT_PORT, FastqQualityTrimWorker::tr("Output File"),
            FastqQualityTrimWorker::tr("Output FASTQ files"));

        QMap<Descriptor, DataTypePtr> inM;
        inM[BaseSlots::URL_SLOT()] = BaseTypes::STRING_TYPE();
        p << new PortDescriptor(inD, DataTypePtr(new MapDataType("cf.input-url", inM)), true);

        QMap<Descriptor, DataTypePtr> outM;
        outM[BaseSlots::URL_SLOT()] = BaseTypes::STRING_TYPE();
        p << new PortDescriptor(outD, DataTypePtr(new MapDataType("cf.output-url", outM)), false, true);
    }

    QList<Attribute*> a;
    {
        Descriptor outDir(BaseNGSWorker::OUT_MODE_ID, FastqQualityTrimWorker::tr("Output folder"),
            FastqQualityTrimWorker::tr("Select an output folder. <b>Custom</b> - specify the output folder in the 'Custom folder' parameter. "
            "<b>Workflow</b> - internal workflow folder. "
            "<b>Input file</b> - the folder of the input file."));

        Descriptor customDir(BaseNGSWorker::CUSTOM_DIR_ID, FastqQualityTrimWorker::tr("Custom folder"),
            FastqQualityTrimWorker::tr("Select the custom output folder."));

        Descriptor outName(BaseNGSWorker::OUT_NAME_ID, FastqQualityTrimWorker::tr("Output file name"),
            FastqQualityTrimWorker::tr("A name of an output file. If default of empty value is provided the output name is the name of the first file with additional extention."));

        Descriptor qualT(QUALITY_ID, FastqQualityTrimWorker::tr("Quality threshold"),
            FastqQualityTrimWorker::tr("Quality threshold for trimming."));

        Descriptor lenT(LEN_ID, FastqQualityTrimWorker::tr("Min Length"),
            FastqQualityTrimWorker::tr("Too short reads are discarded by the filter."));

        Descriptor bothD(BOTH_ID, FastqQualityTrimWorker::tr("Trim both ends"),
            FastqQualityTrimWorker::tr("Trim the both ends of a read or not. Usually, you need to set <b>True</b> for <b>Sanger</b> sequencing and <b>False</b> for <b>NGS</b>"));

        a << new Attribute(outDir, BaseTypes::NUM_TYPE(), false, QVariant(FileAndDirectoryUtils::WORKFLOW_INTERNAL));
        Attribute* customDirAttr = new Attribute(customDir, BaseTypes::STRING_TYPE(), false, QVariant(""));
        customDirAttr->addRelation(new VisibilityRelation(BaseNGSWorker::OUT_MODE_ID, FileAndDirectoryUtils::CUSTOM));
        a << customDirAttr;
        a << new Attribute( outName, BaseTypes::STRING_TYPE(), false, QVariant(BaseNGSWorker::DEFAULT_NAME));
        a << new Attribute( qualT, BaseTypes:: NUM_TYPE(), false, QVariant(30));
        a << new Attribute( lenT, BaseTypes::NUM_TYPE(), false, QVariant(0));
        a << new Attribute( bothD, BaseTypes::BOOL_TYPE(), false, true);
    }

    QMap<QString, PropertyDelegate*> delegates;
    {
        QVariantMap directoryMap;
        QString fileDir = FastqQualityTrimWorker::tr("Input file");
        QString workflowDir = FastqQualityTrimWorker::tr("Workflow");
        QString customD = FastqQualityTrimWorker::tr("Custom");
        directoryMap[fileDir] = FileAndDirectoryUtils::FILE_DIRECTORY;
        directoryMap[workflowDir] = FileAndDirectoryUtils::WORKFLOW_INTERNAL;
        directoryMap[customD] = FileAndDirectoryUtils::CUSTOM;
        delegates[BaseNGSWorker::OUT_MODE_ID] = new ComboBoxDelegate(directoryMap);

        delegates[BaseNGSWorker::CUSTOM_DIR_ID] = new URLDelegate("", "", false, true);

        QVariantMap len; len["minimum"] = 0; len["maximum"] = INT_MAX;
        delegates[QUALITY_ID] = new SpinBoxDelegate(len);
        delegates[LEN_ID] = new SpinBoxDelegate(len);
    }

    ActorPrototype* proto = new IntegralBusActorPrototype(desc, p, a);
    proto->setEditor(new DelegateEditor(delegates));
    proto->setPrompter(new FastqQualityTrimPrompter());

    WorkflowEnv::getProtoRegistry()->registerProto(BaseActorCategories::CATEGORY_NGS_BASIC(), proto);
    DomainFactory *localDomain = WorkflowEnv::getDomainRegistry()->getById(LocalDomainFactory::ID);
    localDomain->registerEntry(new FastqQualityTrimWorkerFactory());
}

/************************************************************************/
/* QualityTrimWorker */
/************************************************************************/
FastqQualityTrimWorker::FastqQualityTrimWorker(Actor *a)
:BaseNGSWorker(a)
{

}

QVariantMap FastqQualityTrimWorker::getCustomParameters() const{
    QVariantMap res;
    res.insert(QUALITY_ID, getValue<int>(QUALITY_ID));
    res.insert(LEN_ID, getValue<int>(LEN_ID));
    res.insert(BOTH_ID, getValue<bool>(BOTH_ID));
    return res;
}

QString FastqQualityTrimWorker::getDefaultFileName() const{
    return ".trimmed.fastq";
}

Task *FastqQualityTrimWorker::getTask(const BaseNGSSetting &settings) const {
    return new FastqQualityTrimTask(settings);
}

//////////////////////////////////////////////////////
//QualityTrimTask
FastqQualityTrimTask::FastqQualityTrimTask(const BaseNGSSetting &settings)
    :BaseNGSTask(settings){

    GCOUNTER(cvar, tvar, "NGS:FASTQQualityTrimmerTask");
}

// diagnose
//   SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS.....................................................
//   ..........................XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX......................
//   ...............................IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII......................
//   .................................JJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJJ......................
//   LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL....................................................
//   !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~
//   |                         |    |        |                              |                     |
//  33                        59   64       73                            104                   126 <- maxValue is value from here
// S 0........................26...31.......40
// X                          -5....0........9.............................40
// I                                0........9.............................40
// J                                   3.....9.............................40
// L 0.2......................26...31........41
//
//  S - Sanger        Phred+33,  raw reads typically (0, 40)
//  X - Solexa        Solexa+64, raw reads typically (-5, 40)
//  I - Illumina 1.3+ Phred+64,  raw reads typically (0, 40)
//  J - Illumina 1.5+ Phred+64,  raw reads typically (3, 40) with 0=unused, 1=unused, 2=Read Segment Quality Control Indicator (bold)
//  L - Illumina 1.8+ Phred+33,  raw reads typically (0, 41)
DNAQualityType FastqQualityTrimTask::detectQualityType(){
    int maxValue = 33;
    int minValue = 126;
    FASTQIterator iter_qual(settings.inputUrl, stateInfo);
    CHECK(!stateInfo.hasError(), DNAQualityType_Sanger);

    int counter = 0;
    while (iter_qual.hasNext()) {
        CHECK(!stateInfo.isCoR(), DNAQualityType_Sanger);

        if (counter > 1000) {   // check only first 1000 reads in file
            break;
        }

        DNASequence dna = iter_qual.next();
        int seqLen = dna.length();
        if (seqLen > dna.quality.qualCodes.length()) {
            continue;
        } else {
            for (int pos = 0; pos <= seqLen - 1; pos++) {
                maxValue = qMax(static_cast<int>(dna.quality.qualCodes.at(pos)), maxValue);
                minValue = qMin(static_cast<int>(dna.quality.qualCodes.at(pos)), minValue);
            }
        }
        counter++;
    }
    return DNAQuality::detectTypeByMinMaxQualityValues(minValue, maxValue);
}

void FastqQualityTrimTask::runStep(){
    int ncount = 0;
    int ycount = 0;

    QScopedPointer<IOAdapter> io(IOAdapterUtils::open(settings.outDir + settings.outName, stateInfo, IOAdapterMode_Append));

    int quality = settings.customParameters.value(QUALITY_ID, 20).toInt();
    int minLen = settings.customParameters.value(LEN_ID, 0).toInt();
    bool bothEnds = settings.customParameters.value(BOTH_ID, false).toInt();
    DNAQualityType qualityType = detectQualityType();
    CHECK_OP(stateInfo, );

    FASTQIterator iter(settings.inputUrl, stateInfo);
    CHECK_OP(stateInfo, );

    while (iter.hasNext()) {
        CHECK_OP(stateInfo, );

        DNASequence dna = iter.next();
        dna.quality.type = qualityType;
        const U2Region acceptedRegion = DNASequenceUtils::trimByQuality(dna, quality, minLen, bothEnds);

        if (0 < acceptedRegion.length) {
            ycount++;
        } else {
            ncount++;
            continue;
        }

        FastqFormat::writeEntry(dna.getName(), dna, io.data(), "Writing error", stateInfo, false);
    }

    algoLog.info(QString("Discarded by trimmer %1").arg(ncount));
    algoLog.info(QString("Accepted by trimmer %1").arg(ycount));
    algoLog.info(QString("Total by trimmer %1").arg(ncount + ycount));
}

QStringList FastqQualityTrimTask::getParameters(U2OpStatus &/*os*/){
    QStringList res;
    return res;
}

///////////////////////////////////////////////////////////////
//MergeFastq
const QString MergeFastqWorkerFactory::ACTOR_ID("MergeFastq");

static const QString INPUT_URLS_ID("input-urls");

/************************************************************************/
/* MergeFastqPrompter */
/************************************************************************/
QString MergeFastqPrompter::composeRichDoc() {
    IntegralBusPort* input = qobject_cast<IntegralBusPort*>(target->getPort(BaseNGSWorker::INPUT_PORT));
    const Actor* producer = input->getProducer(BaseSlots::URL_SLOT().getId());
    QString unsetStr = "<font color='red'>"+tr("unset")+"</font>";
    QString producerName = tr(" from <u>%1</u>").arg(producer ? producer->getLabel() : unsetStr);

    QString doc = tr("Merges input sequences %1.").arg(producerName);
    return doc;
}

/************************************************************************/
/* MergeFastqWorkerFactory */
/************************************************************************/
void MergeFastqWorkerFactory::init() {
    Descriptor desc( ACTOR_ID, MergeFastqWorker::tr("FASTQ Merger"),
        MergeFastqWorker::tr("Merges input sequences to one output file") );

    QList<PortDescriptor*> p;
    {
        Descriptor inD(BaseNGSWorker::INPUT_PORT, MergeFastqWorker::tr("Input File"),
            MergeFastqWorker::tr("Set of FASTQ reads files"));
        Descriptor outD(BaseNGSWorker::OUTPUT_PORT, MergeFastqWorker::tr("Output File"),
            MergeFastqWorker::tr("Output FASTQ file"));

        QMap<Descriptor, DataTypePtr> inM;
        inM[BaseSlots::URL_SLOT()] = BaseTypes::STRING_TYPE();
        p << new PortDescriptor(inD, DataTypePtr(new MapDataType("cf.input-url", inM)), true);

        QMap<Descriptor, DataTypePtr> outM;
        outM[BaseSlots::URL_SLOT()] = BaseTypes::STRING_TYPE();
        p << new PortDescriptor(outD, DataTypePtr(new MapDataType("cf.output-url", outM)), false, true);
    }

    QList<Attribute*> a;
    {
        Descriptor outDir(BaseNGSWorker::OUT_MODE_ID, MergeFastqWorker::tr("Output folder"),
            MergeFastqWorker::tr("Select an output folder. <b>Custom</b> - specify the output folder in the 'Custom folder' parameter. "
            "<b>Workflow</b> - internal workflow folder. "
            "<b>Input file</b> - the folder of the input file."));

        Descriptor customDir(BaseNGSWorker::CUSTOM_DIR_ID, MergeFastqWorker::tr("Custom folder"),
            MergeFastqWorker::tr("Select the custom output folder."));

        Descriptor outName(BaseNGSWorker::OUT_NAME_ID, MergeFastqWorker::tr("Output file name"),
            MergeFastqWorker::tr("A name of an output file. If default of empty value is provided the output name is the name of the first file with additional extention."));


        a << new Attribute(outDir, BaseTypes::NUM_TYPE(), false, QVariant(FileAndDirectoryUtils::WORKFLOW_INTERNAL));
        Attribute* customDirAttr = new Attribute(customDir, BaseTypes::STRING_TYPE(), false, QVariant(""));
        customDirAttr->addRelation(new VisibilityRelation(BaseNGSWorker::OUT_MODE_ID, FileAndDirectoryUtils::CUSTOM));
        a << customDirAttr;
        a << new Attribute( outName, BaseTypes::STRING_TYPE(), false, QVariant(BaseNGSWorker::DEFAULT_NAME));
    }

    QMap<QString, PropertyDelegate*> delegates;
    {
        QVariantMap directoryMap;
        QString fileDir = MergeFastqWorker::tr("Input file");
        QString workflowDir = MergeFastqWorker::tr("Workflow");
        QString customD = MergeFastqWorker::tr("Custom");
        directoryMap[fileDir] = FileAndDirectoryUtils::FILE_DIRECTORY;
        directoryMap[workflowDir] = FileAndDirectoryUtils::WORKFLOW_INTERNAL;
        directoryMap[customD] = FileAndDirectoryUtils::CUSTOM;
        delegates[BaseNGSWorker::OUT_MODE_ID] = new ComboBoxDelegate(directoryMap);

        delegates[BaseNGSWorker::CUSTOM_DIR_ID] = new URLDelegate("", "", false, true);

    }

    ActorPrototype* proto = new IntegralBusActorPrototype(desc, p, a);
    proto->setEditor(new DelegateEditor(delegates));
    proto->setPrompter(new MergeFastqPrompter());

    WorkflowEnv::getProtoRegistry()->registerProto(BaseActorCategories::CATEGORY_NGS_BASIC(), proto);
    DomainFactory *localDomain = WorkflowEnv::getDomainRegistry()->getById(LocalDomainFactory::ID);
    localDomain->registerEntry(new MergeFastqWorkerFactory());
}

/************************************************************************/
/* MergeFastqWorker */
/************************************************************************/
MergeFastqWorker::MergeFastqWorker(Actor *a)
:BaseNGSWorker(a)
{

}

Task * MergeFastqWorker::tick() {
    while (inputUrlPort->hasMessage()) {
        const QString url = takeUrl();
        CHECK(!url.isEmpty(), NULL);
        inputUrls.append(url);
    }
    if (!inputUrlPort->isEnded()) {
        return NULL;
    }

    if(!inputUrls.isEmpty()){
        const QString outputDir = FileAndDirectoryUtils::createWorkingDir(inputUrls.first(), getValue<int>(OUT_MODE_ID), getValue<QString>(CUSTOM_DIR_ID), context->workingDir());

        BaseNGSSetting setting;
        setting.outDir = outputDir;
        setting.outName = getTargetName(inputUrls.first(), outputDir);
        setting.inputUrl = inputUrls.first();
        setting.customParameters = getCustomParameters();
        setting.listeners = createLogListeners();
        Task *t = getTask(setting);
        connect(new TaskSignalMapper(t), SIGNAL(si_taskFinished(Task*)), SLOT(sl_taskFinished(Task*)));
        inputUrls.clear();;
        return t;
    }

    if (inputUrlPort->isEnded()) {
        setDone();
        outputUrlPort->setEnded();
    }
    return NULL;
}

QVariantMap MergeFastqWorker::getCustomParameters() const{
    QVariantMap res;
    res.insert(INPUT_URLS_ID, inputUrls.join(","));
    return res;
}

QString MergeFastqWorker::getDefaultFileName() const{
    return ".merged.fastq";
}

Task *MergeFastqWorker::getTask(const BaseNGSSetting &settings) const {
    return new MergeFastqTask(settings);
}

//////////////////////////////////////////////////////
//MergeFastqTask
MergeFastqTask::MergeFastqTask(const BaseNGSSetting &settings)
    :BaseNGSTask(settings){

    GCOUNTER(cvar, tvar, "NGS:FASTQMergeFastqmerTask");
}

void MergeFastqTask::runStep(){
    QScopedPointer<IOAdapter> io  (IOAdapterUtils::open(settings.outDir + settings.outName, stateInfo, IOAdapterMode_Append));

    QStringList urls = settings.customParameters.value(INPUT_URLS_ID, "").toString().split(",");
    qint64 numberOfSeqs = 0;
    qint64 numberOfFiles = 0;

    foreach (QString url, urls){
        FASTQIterator iter(url, stateInfo);
        if (stateInfo.hasError()) {
            return;
        }
        while(iter.hasNext()){
            if(stateInfo.isCoR()){
                return;
            }
            DNASequence dna = iter.next();
            FastqFormat::writeEntry(dna.getName(), dna, io.data(), "Writing error", stateInfo, false);
            numberOfSeqs++;
        }
        numberOfFiles++;

    }
    algoLog.info(QString("Sequences merged %1").arg(numberOfSeqs));
    algoLog.info(QString("Files merged %1").arg(numberOfFiles));
}

QStringList MergeFastqTask::getParameters(U2OpStatus &/*os*/){
    QStringList res;
    return res;
}

} //LocalWorkflow
} //U2