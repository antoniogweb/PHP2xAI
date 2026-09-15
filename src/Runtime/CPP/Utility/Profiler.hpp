#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace PHP2xAI::Runtime::CPP
{
	struct ProfileEntry
	{
		std::uint64_t calls = 0;
		double totalMs = 0.0;

		double averageMs() const
		{
			return calls > 0 ? totalMs / static_cast<double>(calls) : 0.0;
		}
	};

	class Profiler
	{
	public:
		void add(const std::string &name, double elapsedMs);
		const std::unordered_map<std::string, ProfileEntry> &getEntries() const;
		double totalMs() const;
		void clear();

	private:
		std::unordered_map<std::string, ProfileEntry> entries_;
	};

	class ProfileTimer
	{
	public:
		ProfileTimer(Profiler &profiler, std::string name);
		~ProfileTimer();

		ProfileTimer(const ProfileTimer &) = delete;
		ProfileTimer &operator=(const ProfileTimer &) = delete;

	private:
		using Clock = std::chrono::steady_clock;

		Profiler &profiler_;
		std::string name_;
		Clock::time_point start_;
	};
}
