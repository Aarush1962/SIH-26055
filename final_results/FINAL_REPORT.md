# Final End-to-End EW Scheduler Report

## Pipeline

Raw HDF5 → in-memory RF environment → frozen recursive belief updater → frozen Neural Thompson scheduler. Oracle is computed independently in memory for post-run evaluation only.

## Aggregate results

| Policy | Micro ISR | Mean delay (s) |
|---|---:|---:|
| Random | 33.33% | 1.3438 |
| Greedy belief | 16.67% | 0.0000 |
| Neural Thompson | 41.67% | 0.8000 |
| Oracle | 100.00% | 0.0063 |

- Neural vs Random relative ISR gain: 25.00%
- Neural / Oracle event capture: 41.67%
- Total transmission events: 24
- Test files: 1

## Per-file results

| file        |   total_events |   random_isr |   greedy_isr |   neural_isr |   oracle_isr |   neural_vs_oracle_capture |   neural_mean_delay_s |   oracle_mean_delay_s |
|:------------|---------------:|-------------:|-------------:|-------------:|-------------:|---------------------------:|----------------------:|----------------------:|
| config_0.h5 |             24 |       0.3333 |       0.1667 |       0.4167 |       1.0000 |                     0.4167 |                0.8000 |                0.0063 |

## Event-duration breakdown

| duration_group   |   events |   neural_intercepted |   oracle_intercepted |   neural_isr |   oracle_isr |
|:-----------------|---------:|---------------------:|---------------------:|-------------:|-------------:|
| 5+ steps         |  14.0000 |              10.0000 |              14.0000 |       0.7143 |       1.0000 |
| 3-4 steps        |   5.0000 |               0.0000 |               5.0000 |       0.0000 |       1.0000 |
| 2 steps          |   4.0000 |               0.0000 |               4.0000 |       0.0000 |       1.0000 |
| 1 step           |   1.0000 |               0.0000 |               1.0000 |       0.0000 |       1.0000 |