#include "Profiler.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void Profiler::add(const std::string &name, double elapsedMs)
	{
		auto &entry = entries_[name];
		++entry.calls;
		entry.totalMs += elapsedMs;
	}

	const std::unordered_map<std::string, ProfileEntry> &Profiler::getEntries() const
	{
		return entries_;
	}

	double Profiler::totalMs() const
	{
		double total = 0.0;
		for (const auto &[name, entry] : entries_)
			total += entry.totalMs;
		return total;
	}

	void Profiler::clear()
	{
		entries_.clear();
	}

	ProfileTimer::ProfileTimer(Profiler &profiler, std::string name)
		: profiler_(profiler), name_(std::move(name)), start_(Clock::now())
	{
	}

	ProfileTimer::~ProfileTimer()
	{
		const auto end = Clock::now();
		const double elapsedMs = std::chrono::duration<double, std::milli>(end - start_).count();
		profiler_.add(name_, elapsedMs);
	}
}
