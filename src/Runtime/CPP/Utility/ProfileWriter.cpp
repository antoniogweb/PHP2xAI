#include "ProfileWriter.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>

namespace PHP2xAI::Runtime::CPP
{
	void ProfileWriter::appendBatch(const Profiler &profiler, const std::string &filename, std::size_t batchIndex)
	{
		std::ofstream file(filename, std::ios::app);
		if (!file)
			throw std::runtime_error("Unable to open profiler output file: " + filename);

		const double total = profiler.totalMs();
		std::vector<std::pair<std::string, ProfileEntry>> entries(
			profiler.getEntries().begin(), profiler.getEntries().end());

		std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
			return a.second.totalMs > b.second.totalMs;
		});

		file << "\n=== Batch " << batchIndex << " ===\n";
		file << std::fixed << std::setprecision(3);
		file << std::left << std::setw(42) << "Operation"
			 << std::right << std::setw(10) << "Calls"
			 << std::setw(15) << "Total ms"
			 << std::setw(15) << "Avg ms"
			 << std::setw(10) << "%" << '\n';
		file << std::string(92, '-') << '\n';

		for (const auto &[name, entry] : entries)
		{
			const double percentage = total > 0.0 ? (entry.totalMs / total) * 100.0 : 0.0;
			file << std::left << std::setw(42) << name
				 << std::right << std::setw(10) << entry.calls
				 << std::setw(15) << entry.totalMs
				 << std::setw(15) << entry.averageMs()
				 << std::setw(9) << percentage << "%\n";
		}

		file << std::string(92, '-') << '\n';
		file << "Total profiled time: " << total << " ms\n";
	}
}
