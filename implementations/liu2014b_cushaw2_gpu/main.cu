#include "SingleEnd.h"
#include "PairedEnd.h"

int main(int argc, char* argv[])
{

	double stime, etime;

	/*get the startup time*/
	stime = Utils::getSysTime();

	Options* options = new Options();

	/*parse the arguments*/
	if (!options->parse(argc, argv))
	{
		options->printUsage();
		return 0;
	}

	Genome* rgenome = new Genome(options->getBwtFileBase().c_str(), 1);
	SAM* sam = new SAM(options, rgenome);

	if (options->isPaired() == false)
	{
		/*for single-end alignment*/
		SingleEnd* single = new SingleEnd(options, rgenome, sam);

		/*run the alignment*/
		single->execute();

		/*release aligner*/
		delete single;
	}
	else
	{
		/*for paired-end alignment*/
		PairedEnd* paired = new PairedEnd(options, rgenome, sam);

		/*run the alignment*/
		paired->execute();

		/*release aligner*/
		delete paired;
	}

	/*release resources*/
	delete sam;
	delete rgenome;

	etime = Utils::getSysTime();
	Utils::log("Overall time: %.2f seconds\n", etime - stime);

	return 0;
}
