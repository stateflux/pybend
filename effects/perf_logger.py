#!/usr/bin/env python3
"""
performance logging utility for tracking execution time of effects
"""

import time
import functools
from collections import defaultdict
import threading
import statistics
from typing import Dict, List, Callable, Any, Optional

class PerfLogger:
    """performance logging utility for tracking execution of effects"""
    
    def __init__(self):
        """initialize the performance logger"""
        # store counts of effect calls
        self.effect_counts: Dict[str, int] = defaultdict(int)
        # store execution times for each effect
        self.effect_times: Dict[str, List[float]] = defaultdict(list)
        # store per-effect total time
        self.effect_total_times: Dict[str, float] = defaultdict(float)
        self.enabled = True
        self._lock = threading.Lock()  # for thread safety
    
    def enable(self) -> None:
        """enable performance logging"""
        self.enabled = True
    
    def disable(self) -> None:
        """disable performance logging"""
        self.enabled = False
    
    def wrap_effect(self, effect_obj: Any) -> Any:
        """wrap an effect object to log all calls to its apply method
        
        Args:
            effect_obj: the effect instance to wrap
            
        Returns:
            the wrapped effect object
        """
        original_apply = effect_obj.apply
        effect_name = effect_obj.name
        
        @functools.wraps(original_apply)
        def wrapped_apply(data, width, height, row_length, bytes_per_pixel, intensity):
            if not self.enabled:
                return original_apply(data, width, height, row_length, bytes_per_pixel, intensity)
            
            with self._lock:
                # increment counter
                self.effect_counts[effect_name] += 1
                
                # measure execution time
                start = time.time()
                result = original_apply(data, width, height, row_length, bytes_per_pixel, intensity)
                duration = (time.time() - start) * 1000  # convert to ms
                
                # store duration
                self.effect_times[effect_name].append(duration)
                self.effect_total_times[effect_name] += duration
                
                # print log message
                print(f"[{effect_name}] ran in {duration:.2f}ms (count: {self.effect_counts[effect_name]})")
                
                if duration > 100:  # adjust threshold as needed
                    print(f"!!!!!!!!!! SLOW EFFECT: [{effect_name}] took {duration:.2f}ms")
                    
                return result
                
        # replace the original apply method with our wrapped version
        effect_obj.apply = wrapped_apply
        return effect_obj
    
    def get_summary(self) -> List[Dict[str, Any]]:
        """get summary of all logged effects"""
        summary = []
        with self._lock:
            for name in self.effect_counts:
                times = self.effect_times[name]
                if not times:  # handle case with no measurements
                    continue
                    
                stats = {
                    "name": name,
                    "count": self.effect_counts[name],
                    "total_time_ms": sum(times),
                    "avg_time_ms": statistics.mean(times) if times else 0,
                    "max_time_ms": max(times) if times else 0,
                    "min_time_ms": min(times) if times else 0
                }
                
                # calculate p95 if we have enough data
                if len(times) >= 20:
                    stats["p95_time_ms"] = statistics.quantiles(sorted(times), n=20)[18]
                else:
                    stats["p95_time_ms"] = max(times) if times else 0
                    
                summary.append(stats)
                
        # sort by total time (most expensive first)
        return sorted(summary, key=lambda x: x["total_time_ms"], reverse=True)
    
    def print_summary(self) -> None:
        """print summary to console"""
        summary = self.get_summary()
        
        if not summary:
            print("no performance data collected yet")
            return
            
        print("\n=== PERFORMANCE SUMMARY ===")
        print(f"{'EFFECT':<30} {'COUNT':>8} {'TOTAL(ms)':>12} {'AVG(ms)':>10} {'MAX(ms)':>10} {'P95(ms)':>10}")
        print("-" * 85)
        
        for item in summary:
            print(f"{item['name']:<30} {item['count']:>8} {item['total_time_ms']:>12.2f} "
                  f"{item['avg_time_ms']:>10.2f} {item['max_time_ms']:>10.2f} {item['p95_time_ms']:>10.2f}")
        
        print("\ntotal execution time by effect (ms):")
        total_time = sum(item['total_time_ms'] for item in summary)
        for item in summary:
            percentage = (item['total_time_ms'] / total_time) * 100 if total_time > 0 else 0
            bar_length = int(percentage / 2)  # scale for display
            bar = '█' * bar_length
            print(f"{item['name']:<30} {item['total_time_ms']:>10.2f}ms {bar} {percentage:>5.1f}%")
    
    def reset(self) -> None:
        """reset all counters"""
        with self._lock:
            self.effect_counts.clear()
            self.effect_times.clear()
            self.effect_total_times.clear()

# create singleton instance
perf_logger = PerfLogger() 