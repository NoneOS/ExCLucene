/**
 * This code is released under the
 * Apache License Version 2.0 http://www.apache.org/licenses/.
 *
 */

#include "Array.h"
#include "CodecFactory.h"
#include "common.h"
#include "compressionstats.h"
#include "DeltaFactory.h"
#include "entropy.h"
#include "IndexInfo.h"
#include "IndexLoader.h"
#include "LinearRegressionFactory.h"
#include "regressionstats.h"
#include "util.h"
#include "ztimer.h"


const std::vector<std::string> datasets = { 
	"bd", "bdibda", "bdr", "bdtmf",
	"gov2", "gov2ibda", "gov2r", "gov2tmf", "gov2url"
};


const std::vector<std::string> preprocessors = {
	"Delta",
	"LR"
};


template <typename container>
void verify(const container &a, const container &b) {
	auto mypair = std::mismatch(a.begin(), a.end(), b.begin());
	if (mypair.first != a.end()) {
		std::cout << "First mismatching pair" << std::endl
			      << "expected: " << *mypair.first << std::endl
				  << "miscalculated: " << *mypair.second << std::endl
				  << "index of them: " << mypair.first - a.begin() << std::endl;
		exit(1);
	}
}


void benchDeltaCompression(const std::string &dataset, const std::vector<Array> &docIDs) {
	// entropy
	std::string filename = "result/" + dataset + "_delta_entropy.txt";
	std::ofstream ofsentropy(filename.c_str());
	ofsentropy << "dataset = " << dataset << std::endl;
	ofsentropy << "minlen = " << MINLENDELTA << std::endl << std::endl;

	// distribution of dgaps
	filename = "result/" + dataset + "_delta_gapstats.txt";
	std::ofstream ofsgapstats(filename.c_str());
	ofsgapstats << "dataset = " << dataset << std::endl;
	ofsgapstats << "minlen = " << MINLENDELTA << std::endl << std::endl;

	// compression and decompression
	filename = "result/" + dataset + "_delta_result.txt";
	std::ofstream ofsresult(filename.c_str());
	ofsresult << "dataset = " << dataset << std::endl;
	ofsresult << "minlen = " << MINLENDELTA << std::endl << std::endl;

	Array in(MAXLEN);
	Array out(MAXLEN);


    WallClockTimer z; // timer
	const uint64_t kListNum = docIDs.size(); // number of postings lists

	for (const auto &delta : DeltaFactory<uint32_t>::allSchemes()) {
		std::cout << delta->name() << std::endl;

		compressionstats compstats(delta->name());
		EntropyRecorder er;
		BitWidthHistoGram histo;

		std::cout << "allocating dgaps array" << std::endl;
		std::vector<Array> dgaps(docIDs); 
		std::cout << "generating dgaps" << std::endl;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// progress
			if (uListIdx % (kListNum / 10) == 0)
				std::cout << 100.0 * uListIdx / kListNum << "%..." << std::endl;

			uint32_t *data = dgaps[uListIdx].data();
			uint64_t nvalue = dgaps[uListIdx].size();

			// check whether the list is sorted
			if (!std::is_sorted(data, data + nvalue, std::less_equal<uint32_t>())) {
				std::cerr << "List #" << uListIdx << " is unsorted!" << std::endl;
				exit(1);
			}

			// skip lists of length less than MINLENDELTA
			if (nvalue < MINLENDELTA) {
				compstats.skippednvalue += nvalue;
				continue;
			}
			compstats.processednvalue += nvalue;
			
			// convert docIDs to dgaps 
			z.reset();
			delta->runDelta(data, nvalue);
			compstats.prepTime += z.split();

			er.eat(data, nvalue);            
			histo.eatIntegers(data, nvalue); 
		}
		er.display(ofsentropy, delta->name()); // entropy
		histo.display(ofsgapstats, delta->name()) << std::endl; // distribution of dgaps


		for (const auto &codecname : CodecFactory::allNames()) { // codec->name() looks ugly
            std::cout << codecname << std::endl;

			compstats.codecs.push_back(codecname);
			std::shared_ptr<IntegerCodec> codec = CodecFactory::getFromName(codecname);

			size_t totalcsz = 0;
			double encodetm = 0, decodetm = 0;
			for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
				// progress
				if (uListIdx % (kListNum / 10) == 0)
					std::cout << 100.0 * uListIdx / kListNum << "%..." << std::endl;

				size_t nvalue = dgaps[uListIdx].size();
				size_t csize = 0; // how many words

				// skip postings of length less than MINLENDELTA
				if (nvalue < MINLENDELTA)
					continue;

				for (size_t nDocIdx = 0; nDocIdx < nvalue; ++nDocIdx)
					in[nDocIdx] = dgaps[uListIdx][nDocIdx];

				// encode
				z.reset();
				codec->encodeArray(in.data(), nvalue, out.data(), csize);
				encodetm += z.split();

				totalcsz += csize;


				// decode
				z.reset();
				codec->decodeArray(out.data(), csize, in.data(), nvalue); // we rely on nvalue to decode
				decodetm += z.split();


				// verify encoding and decoding
				verify(dgaps[uListIdx], in);
			}
			std::cout << std::endl;

			compstats.encodeTime.push_back(encodetm);
			compstats.decodeTime.push_back(decodetm);
			compstats.totalcsize.push_back(totalcsz);
		}

		// reconstruct docIDs from dgaps
		std::cout << "verifying deltaing and prefixsuming" << std::endl;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// progress
			if (uListIdx % (kListNum / 10) == 0)
				std::cout << 100.0 * uListIdx / kListNum << "%..." << std::endl;

			uint32_t *data = dgaps[uListIdx].data();
			uint64_t nvalue = dgaps[uListIdx].size();

			if (nvalue < MINLENDELTA)
				continue;

			z.reset();
			delta->runPrefixSum(data, nvalue);
			compstats.postpTime += z.split();

			verify(docIDs[uListIdx], dgaps[uListIdx]);
		}
		
		std::cout << "displaying results" << std::endl;
		compstats.display(ofsresult) << std::endl;

		std::cout << std::endl;
	}
}


void benchLRCompression(const std::string &dataset, const std::vector<Array> &docIDs) {
	// entropy
	std::string filename = "result/" + dataset + "_LR_entropy.txt";
	std::ofstream ofsentropy(filename.c_str(), std::ios::app);
	ofsentropy << "dataset = " << dataset << std::endl;
	ofsentropy << "minlen = " << MINLENLR << std::endl << std::endl;

	// distribution of VDs 
	filename = "result/" + dataset + "_LR_vdstats.txt";
	std::ofstream ofsvdstats(filename.c_str(), std::ios::app);
	ofsvdstats << "dataset = " << dataset << std::endl;
	ofsvdstats << "minlen = " << MINLENLR << std::endl << std::endl;

	// regression stats info
	filename = "result/" + dataset + "_LR_regressionstatinfo.txt";
	std::ofstream ofsregstatinfo(filename.c_str(), std::ios::app);
	ofsregstatinfo << "dataset = " << dataset << std::endl;
	ofsregstatinfo << "minlen = " << MINLENLR << std::endl << std::endl;

	// compression and decompression
	filename = "result/" + dataset + "_LR_result.txt";
	std::ofstream ofsresult(filename.c_str(), std::ios::app);
	ofsresult << "dataset = " << dataset << std::endl;
	ofsresult << "minlen = " << MINLENLR << std::endl << std::endl;

	Array in(MAXLEN);
	Array out(MAXLEN);

    WallClockTimer z; // timer
	const uint64_t kListNum = docIDs.size(); // number of postings lists

	for (const auto &LR : LRFactory::allSchemes()) {
		std::cout << LR[0]->name() << std::endl;

		compressionstats compstats(LR[0]->name());
		EntropyRecorder er;
		BitWidthHistoGram histo;
		regressionstats<100 * 1000> regstats; // default interval size is 100K

		std::cout << "allocating VDs array" << std::endl;
		std::vector<Array> VDs(docIDs); 
		std::cout << "generating VDs" << std::endl;
		uint64_t uLRProcessedListIdx = 0;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// progress
			if (uListIdx % (kListNum / 10) == 0)
				std::cout << 100.0 * uListIdx / kListNum << "%..." << std::endl;

			uint32_t *data = VDs[uListIdx].data();
			uint64_t nvalue = VDs[uListIdx].size();

			// check whether the list is sorted
			if (!std::is_sorted(data, data + nvalue, std::less_equal<uint32_t>())) {
				std::cerr << "List #" << uListIdx << " is unsorted!" << std::endl;
				exit(1);
			}

			// skip lists of length less than MINLENLR
			if (nvalue < MINLENLR) {
				compstats.skippednvalue += nvalue;
				continue;
			}
			compstats.processednvalue += nvalue;
			
			// convert docIDs to VDs 
			z.reset();
			LR[uLRProcessedListIdx]->runConversion(data, nvalue);
			compstats.prepTime += z.split();

			er.eat(data, nvalue);            
			histo.eatIntegers(data, nvalue); 
			regstats.accumulate(nvalue, LR[uLRProcessedListIdx]->statInfo);

			++uLRProcessedListIdx;
		}
		assert(uLRProcessedListIdx < MAXNUMLR);
		er.display(ofsentropy, LR[0]->name()); // entropy
		histo.display(ofsvdstats, LR[0]->name()) << std::endl; // distribution of VDs
		regstats.display(ofsregstatinfo, LR[0]->name()) << std::endl; // regression stat info

		for (const auto &codecname : CodecFactory::allNames()) { // codec->name() looks ugly
			std::cout << codecname << std::endl;

			std::shared_ptr<IntegerCodec> codec = CodecFactory::getFromName(codecname);
			compstats.codecs.push_back(codecname);

			size_t totalcsz = 0;
			double encodetm = 0, decodetm = 0;
			uLRProcessedListIdx = 0;
			for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
				// progress
				if (uListIdx % (kListNum / 10) == 0)
					std::cout << 100.0 * uListIdx / kListNum << "%..." << std::endl;

				size_t nvalue = VDs[uListIdx].size();
				size_t csize = 0; // how many words

				// skip lists of length less than MINLENLR
				if (nvalue < MINLENLR)
					continue;

				for (size_t nDocIdx = 0; nDocIdx < nvalue; ++nDocIdx)
					in[nDocIdx] = VDs[uListIdx][nDocIdx];

				// encode
				z.reset();
				codec->encodeArray(in.data(), nvalue, out.data(), csize);
				encodetm += z.split();

				totalcsz += csize;
				totalcsz += LR[uLRProcessedListIdx++]->sizeOfAuxiliaryInfo(); // specific to LR-based compression


				// decode
				z.reset();
				codec->decodeArray(out.data(), csize, in.data(), nvalue); // we rely on nvalue to decode
				decodetm += z.split();


				// verify encoding and decoding
				verify(VDs[uListIdx], in);
			}
			std::cout << std::endl;

			compstats.encodeTime.push_back(encodetm);
			compstats.decodeTime.push_back(decodetm);
			compstats.totalcsize.push_back(totalcsz);
		}

		// reconstruct docIDs from VDs
		std::cout << "verifying converting and reconstructing" << std::endl;
		uLRProcessedListIdx = 0;
		for (uint64_t uListIdx = 0; uListIdx < kListNum; ++uListIdx) {
			// progress
			if (uListIdx % (kListNum / 10) == 0)
				std::cout << 100.0 * uListIdx / kListNum << "%..." << std::endl;

			uint32_t *data = VDs[uListIdx].data();
			uint64_t nvalue = VDs[uListIdx].size();

			if (nvalue < MINLENLR)
				continue;

			z.reset();
			LR[uLRProcessedListIdx]->runReconstruction(data, nvalue);
			compstats.postpTime += z.split();

			++uLRProcessedListIdx;

			verify(docIDs[uListIdx], VDs[uListIdx]);
		}
		
		std::cout << "displaying results" << std::endl;
		compstats.display(ofsresult) << std::endl;

		std::cout << std::endl;
	}
}


void displayUsage() {
	std::cout << "run as './benCompression dataset preprocessor'" << std::endl;
	std::cout << "where datasets are" << std::endl;
	for (auto &dataset : datasets) {
		std::cout << dataset << std::endl;
	}

	std::cout << "preprocessors are" << std::endl;
	for (auto &preprocessor : preprocessors) {
		std::cout << preprocessor << std::endl;
	}
}


int main(int argc, char **argv) {
	if (argc < 3) {
		displayUsage();
		return -1;
	}

	std::string dataset = argv[1];
	std::string indexDir = "path_index" + dataset + "/"; // this is the directory of your index files
	std::cout << dataset << std::endl;

	// load index
	IndexLoader idxLoader(indexDir, dataset);
	idxLoader.loadIndex();
	const std::vector<Array> &docIDs = idxLoader.postings;

	std::string preprocessor = argv[2];
	if (preprocessor == "Delta") {
		std::cout << "benchmarking Delta Compression" << std::endl;
		benchDeltaCompression(dataset, docIDs);
	}
	else if (preprocessor == "LR") {
		std::cout << "benchmarking LR Compression" << std::endl;
		benchLRCompression(dataset, docIDs);
	}
	else { // run both
		std::cout << "benchmarking Delta Compression" << std::endl;
		benchDeltaCompression(dataset, docIDs);
		std::cout << "benchmarking LR Compression" << std::endl;
		benchLRCompression(dataset, docIDs);
	}

	return 0;
}

