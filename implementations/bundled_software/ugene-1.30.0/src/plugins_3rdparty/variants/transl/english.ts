<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE TS>
<TS version="2.1" language="en_US" sourcelanguage="en">
<context>
    <name>U2::LocalWorkflow::CallVariantsPrompter</name>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="690"/>
        <source>unset</source>
        <translation>unset</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="698"/>
        <source>For reference sequence from &lt;u&gt;%1&lt;/u&gt;,</source>
        <translation>For reference sequence from &lt;u&gt;%1&lt;/u&gt;,</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="701"/>
        <source>with assembly data provided by &lt;u&gt;%1&lt;/u&gt;</source>
        <translation>with assembly data provided by &lt;u&gt;%1&lt;/u&gt;</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="703"/>
        <source>%1 call variants %2.</source>
        <translation>%1 call variants %2.</translation>
    </message>
</context>
<context>
    <name>U2::LocalWorkflow::CallVariantsTask</name>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="52"/>
        <source>Call variants for %1</source>
        <translation>Call variants for %1</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="65"/>
        <source>reference</source>
        <translation>reference</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="67"/>
        <source>assembly</source>
        <translation>assembly</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="75"/>
        <source>The %1 file does not exist: %2</source>
        <translation>The %1 file does not exist: %2</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="88"/>
        <source>No assembly files</source>
        <translation>No assembly files</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="93"/>
        <source>No dbi storage</source>
        <translation>No dbi storage</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="97"/>
        <source>No sequence URL</source>
        <translation>No sequence URL</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="135"/>
        <source>No document loaded</source>
        <translation>No document loaded</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="139"/>
        <source>Incorrect variant track object in %1</source>
        <translation>Incorrect variant track object in %1</translation>
    </message>
</context>
<context>
    <name>U2::LocalWorkflow::CallVariantsWorker</name>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="135"/>
        <source>Empty input slot: %1</source>
        <translation>Empty input slot: %1</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="155"/>
        <source>Input sequences</source>
        <translation>Input sequences</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="156"/>
        <source>A nucleotide reference sequence.</source>
        <translation>A nucleotide reference sequence.</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="163"/>
        <source>Input assembly</source>
        <translation>Input assembly</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="164"/>
        <source>Position sorted alignment file</source>
        <translation>Position sorted alignment file</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="171"/>
        <source>Output variations</source>
        <translation>Output variations</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="172"/>
        <source>Output tracks with SNPs and short INDELs</source>
        <translation>Output tracks with SNPs and short INDELs</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="177"/>
        <source>Call Variants with SAMtools</source>
        <translation>Call Variants with SAMtools</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="178"/>
        <source>Calls SNPs and INDELS with SAMtools mpileup and bcftools.</source>
        <translation>Calls SNPs and INDELS with SAMtools mpileup and bcftools.</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="186"/>
        <source>Output variants file</source>
        <translation>Output variants file</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="187"/>
        <source>The url to the file with the extracted variations.</source>
        <translation>The url to the file with the extracted variations.</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="190"/>
        <source>Use reference from</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="191"/>
        <source>&lt;p&gt;Specify &quot;File&quot; to set a single reference sequence for all input NGS assemblies. The reference should be set in the &quot;Reference&quot; parameter.&lt;/p&gt;&lt;p&gt;Specify &quot;Input port&quot; to be able to set different references for difference NGS assemblies. The references should be input via the &quot;Input sequences&quot; port (e.g. use datasets in the &quot;Read Sequence&quot; element).&lt;/p&gt;</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="197"/>
        <source>Reference</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="198"/>
        <source>&lt;p&gt;Specify a file with the reference sequence.&lt;/p&gt;&lt;p&gt;The sequence will be used as reference for all datasets with NGS assemblies.&lt;/p&gt;</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="202"/>
        <source>Illumina-1.3+ encoding</source>
        <translation>Illumina-1.3+ encoding</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="203"/>
        <source>Assume the quality is in the Illumina 1.3+ encoding (mpileup)(-6).</source>
        <translation>Assume the quality is in the Illumina 1.3+ encoding (mpileup)(-6).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="206"/>
        <source>Count anomalous read pairs</source>
        <translation>Count anomalous read pairs</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="207"/>
        <source>Do not skip anomalous read pairs in variant calling(mpileup)(-A).</source>
        <translation>Do not skip anomalous read pairs in variant calling(mpileup)(-A).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="210"/>
        <source>Disable BAQ computation</source>
        <translation>Disable BAQ computation</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="211"/>
        <source>Disable probabilistic realignment for the computation of base alignment quality (BAQ). BAQ is the Phred-scaled probability of a read base being misaligned. Applying this option greatly helps to reduce false SNPs caused by misalignments. (mpileup)(-B).</source>
        <translation>Disable probabilistic realignment for the computation of base alignment quality (BAQ). BAQ is the Phred-scaled probability of a read base being misaligned. Applying this option greatly helps to reduce false SNPs caused by misalignments. (mpileup)(-B).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="216"/>
        <source>Mapping quality downgrading coefficient</source>
        <translation>Mapping quality downgrading coefficient</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="217"/>
        <source>Coefficient for downgrading mapping quality for reads containing excessive mismatches. Given a read with a phred-scaled mapping quality q of being generated from the mapped position, the new mapping quality is about sqrt((INT-q)/INT)*INT. A zero value disables this functionality; if enabled, the recommended value for BWA is 50 (mpileup)(-C).</source>
        <translation>Coefficient for downgrading mapping quality for reads containing excessive mismatches. Given a read with a phred-scaled mapping quality q of being generated from the mapped position, the new mapping quality is about sqrt((INT-q)/INT)*INT. A zero value disables this functionality; if enabled, the recommended value for BWA is 50 (mpileup)(-C).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="222"/>
        <source>Max number of reads per input BAM</source>
        <translation>Max number of reads per input BAM</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="223"/>
        <source>At a position, read maximally the number of reads per input BAM (mpileup)(-d).</source>
        <translation>At a position, read maximally the number of reads per input BAM (mpileup)(-d).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="226"/>
        <source>Extended BAQ computation</source>
        <translation>Extended BAQ computation</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="227"/>
        <source>Extended BAQ computation. This option helps sensitivity especially for MNPs, but may hurt specificity a little bit (mpileup)(-E).</source>
        <translation>Extended BAQ computation. This option helps sensitivity especially for MNPs, but may hurt specificity a little bit (mpileup)(-E).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="231"/>
        <source>BED or position list file</source>
        <translation>BED or position list file</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="232"/>
        <source>BED or position list file containing a list of regions or sites where pileup or BCF should be generated (mpileup)(-l).</source>
        <translation>BED or position list file containing a list of regions or sites where pileup or BCF should be generated (mpileup)(-l).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="236"/>
        <source>Pileup region</source>
        <translation>Pileup region</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="237"/>
        <source>Only generate pileup in region STR (mpileup)(-r).</source>
        <translation>Only generate pileup in region STR (mpileup)(-r).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="240"/>
        <source>Minimum mapping quality</source>
        <translation>Minimum mapping quality</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="241"/>
        <source>Minimum mapping quality for an alignment to be used (mpileup)(-q).</source>
        <translation>Minimum mapping quality for an alignment to be used (mpileup)(-q).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="244"/>
        <source>Minimum base quality</source>
        <translation>Minimum base quality</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="245"/>
        <source>Minimum base quality for a base to be considered (mpileup)(-Q).</source>
        <translation>Minimum base quality for a base to be considered (mpileup)(-Q).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="248"/>
        <source>Gap extension error</source>
        <translation>Gap extension error</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="249"/>
        <source>Phred-scaled gap extension sequencing error probability. Reducing INT leads to longer indels (mpileup)(-e).</source>
        <translation>Phred-scaled gap extension sequencing error probability. Reducing INT leads to longer indels (mpileup)(-e).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="252"/>
        <source>Homopolymer errors coefficient</source>
        <translation>Homopolymer errors coefficient</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="253"/>
        <source>Coefficient for modeling homopolymer errors. Given an l-long homopolymer run, the sequencing error of an indel of size s is modeled as INT*s/l (mpileup)(-h).</source>
        <translation>Coefficient for modeling homopolymer errors. Given an l-long homopolymer run, the sequencing error of an indel of size s is modeled as INT*s/l (mpileup)(-h).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="256"/>
        <source>No INDELs</source>
        <translation>No INDELs</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="257"/>
        <source>Do not perform INDEL calling (mpileup)(-I).</source>
        <translation>Do not perform INDEL calling (mpileup)(-I).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="260"/>
        <source>Max INDEL depth</source>
        <translation>Max INDEL depth</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="261"/>
        <source>Skip INDEL calling if the average per-sample depth is above INT (mpileup)(-L).</source>
        <translation>Skip INDEL calling if the average per-sample depth is above INT (mpileup)(-L).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="264"/>
        <source>Gap open error</source>
        <translation>Gap open error</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="265"/>
        <source>Phred-scaled gap open sequencing error probability. Reducing INT leads to more indel calls (mpileup)(-o).</source>
        <translation>Phred-scaled gap open sequencing error probability. Reducing INT leads to more indel calls (mpileup)(-o).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="268"/>
        <source>List of platforms for indels</source>
        <translation>List of platforms for indels</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="269"/>
        <source>Comma dilimited list of platforms (determined by @RG-PL) from which indel candidates are obtained.It is recommended to collect indel candidates from sequencing technologies that have low indel error rate such as ILLUMINA (mpileup)(-P).</source>
        <translation>Comma dilimited list of platforms (determined by @RG-PL) from which indel candidates are obtained.It is recommended to collect indel candidates from sequencing technologies that have low indel error rate such as ILLUMINA (mpileup)(-P).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="274"/>
        <source>Retain all possible alternate</source>
        <translation>Retain all possible alternate</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="275"/>
        <source>Retain all possible alternate alleles at variant sites. By default, the view command discards unlikely alleles (bcf view)(-A).</source>
        <translation>Retain all possible alternate alleles at variant sites. By default, the view command discards unlikely alleles (bcf view)(-A).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="278"/>
        <source>Indicate PL</source>
        <translation>Indicate PL</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="279"/>
        <source>Indicate PL is generated by r921 or before (ordering is different) (bcf view)(-F).</source>
        <translation>Indicate PL is generated by r921 or before (ordering is different) (bcf view)(-F).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="282"/>
        <source>No genotype information</source>
        <translation>No genotype information</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="283"/>
        <source>Suppress all individual genotype information (bcf view)(-G).</source>
        <translation>Suppress all individual genotype information (bcf view)(-G).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="286"/>
        <source>A/C/G/T only</source>
        <translation>A/C/G/T only</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="287"/>
        <source>Skip sites where the REF field is not A/C/G/T (bcf view)(-N).</source>
        <translation>Skip sites where the REF field is not A/C/G/T (bcf view)(-N).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="290"/>
        <source>List of sites</source>
        <translation>List of sites</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="291"/>
        <source>List of sites at which information are outputted (bcf view)(-l).</source>
        <translation>List of sites at which information are outputted (bcf view)(-l).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="294"/>
        <source>QCALL likelihood</source>
        <translation>QCALL likelihood</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="295"/>
        <source>Output the QCALL likelihood format (bcf view)(-Q).</source>
        <translation>Output the QCALL likelihood format (bcf view)(-Q).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="298"/>
        <source>List of samples</source>
        <translation>List of samples</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="299"/>
        <source>List of samples to use. The first column in the input gives the sample names and the second gives the ploidy, which can only be 1 or 2. When the 2nd column is absent, the sample ploidy is assumed to be 2. In the output, the ordering of samples will be identical to the one in FILE (bcf view)(-s).</source>
        <translation>List of samples to use. The first column in the input gives the sample names and the second gives the ploidy, which can only be 1 or 2. When the 2nd column is absent, the sample ploidy is assumed to be 2. In the output, the ordering of samples will be identical to the one in FILE (bcf view)(-s).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="305"/>
        <source>Min samples fraction</source>
        <translation>Min samples fraction</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="306"/>
        <source>skip loci where the fraction of samples covered by reads is below FLOAT (bcf view)(-d).</source>
        <translation>skip loci where the fraction of samples covered by reads is below FLOAT (bcf view)(-d).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="309"/>
        <source>Per-sample genotypes</source>
        <translation>Per-sample genotypes</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="310"/>
        <source>Call per-sample genotypes at variant sites (bcf view)(-g).</source>
        <translation>Call per-sample genotypes at variant sites (bcf view)(-g).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="313"/>
        <source>INDEL-to-SNP Ratio</source>
        <translation>INDEL-to-SNP Ratio</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="314"/>
        <source>Ratio of INDEL-to-SNP mutation rate (bcf view)(-i).</source>
        <translation>Ratio of INDEL-to-SNP mutation rate (bcf view)(-i).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="317"/>
        <source>Max P(ref|D)</source>
        <translation>Max P(ref|D)</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="318"/>
        <source>A site is considered to be a variant if P(ref|D)&lt;FLOAT (bcf view)(-p).</source>
        <translation>A site is considered to be a variant if P(ref|D) less than FLOAT (bcf view)(-p).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="321"/>
        <source>Prior allele frequency spectrum</source>
        <translation>Prior allele frequency spectrum</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="322"/>
        <source>If STR can be full, cond2, flat or the file consisting of error output from a previous variant calling run (bcf view)(-P).</source>
        <translation>If STR can be full, cond2, flat or the file consisting of error output from a previous variant calling run (bcf view)(-P).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="325"/>
        <source>Mutation rate</source>
        <translation>Mutation rate</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="326"/>
        <source>Scaled mutation rate for variant calling (bcf view)(-t).</source>
        <translation>Scaled mutation rate for variant calling (bcf view)(-t).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="329"/>
        <source>Pair/trio calling</source>
        <translation>Pair/trio calling</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="330"/>
        <source>Enable pair/trio calling. For trio calling, option -s is usually needed to be applied to configure the trio members and their ordering. In the file supplied to the option -s, the first sample must be the child, the second the father and the third the mother. The valid values of STR are &apos;pair&apos;, &apos;trioauto&apos;, &apos;trioxd&apos; and &apos;trioxs&apos;, where &apos;pair&apos; calls differences between two input samples, and &apos;trioxd&apos; (&apos;trioxs&apos;)specifies that the input is from the X chromosome non-PAR regions and the child is a female (male) (bcf view)(-T).</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <source>Enable pair/trio calling. For trio calling, option -s is usually needed to be applied to configure the trio members and their ordering. In the file supplied to the option -s, the first sample must be the child, the second the father and the third the mother. The valid values of STR are &apos;ÂpairÂ&apos;, &apos;ÂtrioautoÂ&apos;, &apos;ÂtrioxdÂ&apos; and &apos;ÂtrioxsÂ&apos;, where &apos;ÂpairÂ&apos; calls differences between two input samples, and &apos;ÂtrioxdÂ&apos; (&apos;ÂtrioxsÂ&apos;)specifies that the input is from the X chromosome non-PAR regions and the child is a female (male) (bcf view)(-T).</source>
        <translation type="vanished">Enable pair/trio calling. For trio calling, option -s is usually needed to be applied to configure the trio members and their ordering. In the file supplied to the option -s, the first sample must be the child, the second the father and the third the mother. The valid values of STR are &apos;ÂpairÂ&apos;, &apos;ÂtrioautoÂ&apos;, &apos;ÂtrioxdÂ&apos; and &apos;ÂtrioxsÂ&apos;, where &apos;ÂpairÂ&apos; calls differences between two input samples, and &apos;ÂtrioxdÂ&apos; (&apos;ÂtrioxsÂ&apos;)specifies that the input is from the X chromosome non-PAR regions and the child is a female (male) (bcf view)(-T).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="337"/>
        <source>N group-1 samples</source>
        <translation>N group-1 samples</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="338"/>
        <source>Number of group-1 samples. This option is used for dividing the samples into two groups for contrast SNP calling or association test. When this option is in use, the followingVCF INFO will be outputted: PC2, PCHI2 and QCHI2 (bcf view)(-1).</source>
        <translation>Number of group-1 samples. This option is used for dividing the samples into two groups for contrast SNP calling or association test. When this option is in use, the followingVCF INFO will be outputted: PC2, PCHI2 and QCHI2 (bcf view)(-1).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="343"/>
        <source>N permutations</source>
        <translation>N permutations</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="344"/>
        <source>Number of permutations for association test (effective only with -1) (bcf view)(-U).</source>
        <translation>Number of permutations for association test (effective only with -1) (bcf view)(-U).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="347"/>
        <source>Max P(chi^2)</source>
        <translation>Max P(chi^2)</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="348"/>
        <source>Only perform permutations for P(chi^2)&lt;FLOAT (N permutations) (bcf view)(-X).</source>
        <translation>Only perform permutations for P(chi^2)&lt;FLOAT (N permutations) (bcf view)(-X).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="352"/>
        <source>Minimum RMS quality</source>
        <translation>Minimum RMS quality</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="353"/>
        <source>Minimum RMS mapping quality for SNPs (varFilter) (-Q).</source>
        <translation>Minimum RMS mapping quality for SNPs (varFilter) (-Q).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="355"/>
        <source>Minimum read depth</source>
        <translation>Minimum read depth</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="356"/>
        <source>Minimum read depth (varFilter) (-d).</source>
        <translation>Minimum read depth (varFilter) (-d).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="358"/>
        <source>Maximum read depth</source>
        <translation>Maximum read depth</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="359"/>
        <source>Maximum read depth (varFilter) (-D).</source>
        <translation>Maximum read depth (varFilter) (-D).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="361"/>
        <source>Alternate bases</source>
        <translation>Alternate bases</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="362"/>
        <source>Minimum number of alternate bases (varFilter) (-a).</source>
        <translation>Minimum number of alternate bases (varFilter) (-a).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="364"/>
        <source>Gap size</source>
        <translation>Gap size</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="365"/>
        <source>SNP within INT bp around a gap to be filtered (varFilter) (-w).</source>
        <translation>SNP within INT bp around a gap to be filtered (varFilter) (-w).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="367"/>
        <source>Window size</source>
        <translation>Window size</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="368"/>
        <source>Window size for filtering adjacent gaps (varFilter) (-W).</source>
        <translation>Window size for filtering adjacent gaps (varFilter) (-W).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="370"/>
        <source>Strand bias</source>
        <translation>Strand bias</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="371"/>
        <source>Minimum P-value for strand bias (given PV4) (varFilter) (-1).</source>
        <translation>Minimum P-value for strand bias (given PV4) (varFilter) (-1).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="373"/>
        <source>BaseQ bias</source>
        <translation>BaseQ bias</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="374"/>
        <source>Minimum P-value for baseQ bias (varFilter) (-2).</source>
        <translation>Minimum P-value for baseQ bias (varFilter) (-2).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="376"/>
        <source>MapQ bias</source>
        <translation>MapQ bias</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="377"/>
        <source>Minimum P-value for mapQ bias (varFilter) (-3).</source>
        <translation>Minimum P-value for mapQ bias (varFilter) (-3).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="379"/>
        <source>End distance bias</source>
        <translation>End distance bias</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="380"/>
        <source>Minimum P-value for end distance bias (varFilter) (-4).</source>
        <translation>Minimum P-value for end distance bias (varFilter) (-4).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="382"/>
        <source>HWE</source>
        <translation>HWE</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="383"/>
        <source>Minimum P-value for HWE (plus F&lt;0) (varFilter) (-e).</source>
        <translation>Minimum P-value for HWE (plus F&lt;0) (varFilter) (-e).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="385"/>
        <source>Log filtered</source>
        <translation>Log filtered</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="386"/>
        <source>Print filtered variants into the log (varFilter) (-p).</source>
        <translation>Print filtered variants into the log (varFilter) (-p).</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="658"/>
        <source>Input port</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="659"/>
        <source>File</source>
        <translation type="unfinished"></translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="842"/>
        <source>Assembly URL slot is empty. Please, specify the URL slot</source>
        <translation>Assembly URL slot is empty. Please, specify the URL slot</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="867"/>
        <source>Ref sequence URL slot is empty. Please, specify the URL slot</source>
        <translation>Ref sequence URL slot is empty. Please, specify the URL slot</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="946"/>
        <source>Not enough references</source>
        <translation>Not enough references</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="951"/>
        <source>The dataset slot is not binded, only the first reference sequence against all assemblies was processed.</source>
        <translation>The dataset slot is not binded, only the first reference sequence against all assemblies was processed.</translation>
    </message>
    <message>
        <location filename="../src/SamtoolMpileupWorker.cpp" line="954"/>
        <source>Not enough assemblies</source>
        <translation>Not enough assemblies</translation>
    </message>
</context>
<context>
    <name>U2::LocalWorkflow::SamtoolsMpileupTask</name>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="161"/>
        <source>Samtool mpileup for %1 </source>
        <translation>Samtool mpileup for %1 </translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="168"/>
        <source>No reference sequence URL to do pileup</source>
        <translation>No reference sequence URL to do pileup</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="173"/>
        <source>No assembly URL to do pileup</source>
        <translation>No assembly URL to do pileup</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="179"/>
        <source>There is an assembly with an empty path</source>
        <translation>There is an assembly with an empty path</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="188"/>
        <source>Can not create the folder: </source>
        <translation>Can not create the folder: </translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="243"/>
        <source>Can not run %1 tool</source>
        <translation>Can not run %1 tool</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="249"/>
        <source>%1 tool exited with code %2</source>
        <translation>%1 tool exited with code %2</translation>
    </message>
    <message>
        <location filename="../src/AssemblySamtoolsMpileup.cpp" line="251"/>
        <source>Tool %1 finished successfully</source>
        <translation>Tool %1 finished successfully</translation>
    </message>
</context>
<context>
    <name>U2::SamtoolsPlugin</name>
    <message>
        <location filename="../src/SamtoolsPlugin.cpp" line="36"/>
        <source>Samtools plugin</source>
        <translation>Samtools plugin</translation>
    </message>
    <message>
        <location filename="../src/SamtoolsPlugin.cpp" line="36"/>
        <source>Samtools plugin for NGS data analysis</source>
        <translation>Samtools plugin for NGS data analysis</translation>
    </message>
</context>
</TS>
