// helper_timer.h — Platform-abstracted timing utility
//
// Inspired by NVIDIA cuda-samples Common/helper_timer.h
// Provides simple start/stop/elapsed timing for benchmark measurements.

#pragma once

#include <cstdint>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/time.h>
#endif

class StopWatch {
public:
    StopWatch() : total_time_(0.0), running_(false), sessions_(0) {}

    void start() {
        get_time(&start_time_);
        running_ = true;
    }

    void stop() {
        if (running_) {
            struct timeval end;
            get_time(&end);
            total_time_ += diff_time(start_time_, end);
            running_ = false;
            sessions_++;
        }
    }

    void reset() {
        total_time_ = 0.0;
        sessions_ = 0;
        running_ = false;
    }

    // Elapsed time in milliseconds
    double elapsed_ms() const {
        if (running_) {
            struct timeval now;
            get_time(&now);
            return total_time_ + diff_time(start_time_, now);
        }
        return total_time_;
    }

    // Average time per session in milliseconds
    double average_ms() const {
        return (sessions_ > 0) ? (total_time_ / sessions_) : 0.0;
    }

    int sessions() const { return sessions_; }

private:
    struct timeval start_time_;
    double total_time_;
    bool running_;
    int sessions_;

    static void get_time(struct timeval* tv) {
#if defined(_WIN32)
        LARGE_INTEGER freq, counter;
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&counter);
        tv->tv_sec = (long)(counter.QuadPart / freq.QuadPart);
        tv->tv_usec = (long)((counter.QuadPart % freq.QuadPart) * 1000000 / freq.QuadPart);
#else
        gettimeofday(tv, nullptr);
#endif
    }

    static double diff_time(const struct timeval& start, const struct timeval& end) {
        return 1000.0 * (end.tv_sec - start.tv_sec)
             + 0.001 * (end.tv_usec - start.tv_usec);
    }
};
