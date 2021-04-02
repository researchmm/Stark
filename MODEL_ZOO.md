# STARK Model Zoo

Here we provide the performance of the STARK trackers on multiple tracking benchmarks and the corresponding raw results. 
The model weights and the corresponding training logs are also given by the links.

## Tracking
### Models & Logs

<table>
  <tr>
    <th>Model</th>
    <th>LaSOT<br>AUC (%)</th>
    <th>GOT-10k<br>AO (%)</th>
    <th>TrackingNet<br>AUC (%)</th>
    <th>VOT2020<br>EAO</th>
    <th>VOT2020-LT<br>F-score (%)</th>
    <th>Models</th>
    <th>Logs</th>
    <th>Logs(GOT10K)</th>
  </tr>
  <tr>
    <td>STARK-S50</td>
    <td>65.8</td>
    <td>67.2</td>
    <td>80.3</td>
    <td>0.462</td>
    <td>-</td>
    <td><a href="https://drive.google.com/drive/folders/1144cEuF_yn9UwTfrSVl5wmaMK3F92q42?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/file/d/1_YI0CX52vg8zN6hWsYK22_78FXPiukdv/view?usp=sharing">log</a></td>
    <td><a href="https://drive.google.com/file/d/1xLUeV9I9tejT4eYd1mYpeB_AsndiaJNI/view?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>STARK-ST50</td>
    <td>66.4</td>
    <td>68.0</td>
    <td>81.3</td>
    <td>0.505</td>
    <td>70.2</td>
    <td><a href="https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/drive/folders/1RcPoBxI1_E6U9s5Y6BEhQH_ov-sT7SJM?usp=sharing">log</a></td>
    <td><a href="https://drive.google.com/drive/folders/13guPF1MUOaRa09_4y_K9do9yhQsC_y_y?usp=sharing">log</a></td>
  </tr>
  <tr>
    <td>STARK-ST101</td>
    <td>67.1</td>
    <td>68.8</td>
    <td>82.0</td>
    <td>0.497</td>
    <td>70.1</td>
    <td><a href="https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN?usp=sharing">model</a></td>
    <td><a href="https://drive.google.com/drive/folders/1nTDRfG0K0w2XiP5RDrYJXhotUYQJBNoY?usp=sharing">log</a></td>
    <td><a href="https://drive.google.com/drive/folders/1PR6PRdARHFKBDSjoqeO7qxx9y87AZWSD?usp=sharing">log</a></td>
  </tr>


</table>

The downloaded checkpoints should be organized in the following structure
   ```
   ${STARK_ROOT}
    -- checkpoints
        -- train
            -- stark_s
                -- baseline
                    STARKS_ep0500.pth.tar
                -- baseline_got10k_only
                    STARKS_ep0500.pth.tar
            -- stark_st2
                -- baseline
                    STARKST_ep0050.pth.tar
                -- baseline_got10k_only
                    STARKST_ep0050.pth.tar
                -- baseline_R101
                    STARKST_ep0050.pth.tar
                -- baseline_R101_got10k_only
                    STARKST_ep0050.pth.tar
   ```
### Raw Results
The [raw results](https://drive.google.com/drive/folders/16rWehrDmJVmvMsadffOPaCczz7pst4PH?usp=sharing) are in the format [top_left_x, top_left_y, width, height]. Raw results of GOT-10K and TrackingNet can be 
directly submitted to the corresponding evaluation servers. The folder ```test/tracking_results/``` contains raw results 
for the LaSOT dataset and results should be organized in the following structure
   ```
   ${STARK_ROOT}
    -- test
        -- tracking_results
            -- stark_s
                -- baseline
                    airplane-1.txt
                    airplane-13.txt
                    ...
            -- stark_st2
                -- baseline
                    airplane-1.txt
                    airplane-13.txt
                    ...
                -- baseline_R101
                    airplane-1.txt
                    airplane-13.txt
                    ...
   ```
The raw results of VOT2020 and VOT2020-LT should be organized in the following structure
   ```
   ${STARK_ROOT}
    -- external
        -- vot20
            -- stark_s50
                -- results
            -- stark_s50_ar
                -- results
            -- stark_st50
                -- results
            -- stark_st50_ar
                -- results
            -- stark_st101
                -- results
            -- stark_st101_ar
                -- results
        -- vot20_lt
            -- stark_st50
                -- results
            -- stark_st101
                -- results
   ```