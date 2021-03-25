# STARK Model Zoo

Here we provide the performance of the STARK trackers on multiple tracking benchmarks and the corresponding raw results. 
The model weights and the corresponding training logs are also given by the links.

## Tracking
### Models

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
  </tr>
  <tr>
    <td>STARK-S50</td>
    <td>65.8</td>
    <td>67.2</td>
    <td>80.3</td>
    <td>0.462</td>
    <td>-</td>
    <td><a href="">model</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <td>STARK-ST50</td>
    <td>66.4</td>
    <td>68.0</td>
    <td>81.3</td>
    <td>0.505</td>
    <td>70.2</td>
    <td><a href="">model</a></td>
    <td><a href="">logs</a></td>
  </tr>
  <tr>
    <td>STARK-ST101</td>
    <td>67.1</td>
    <td>68.8</td>
    <td>82.0</td>
    <td>0.497</td>
    <td>70.1</td>
    <td><a href="">model</a></td>
    <td><a href="">logs</a></td>
  </tr>


</table>

### Raw Results
The raw results are in the format [top_left_x, top_left_y, width, height].
The folder ```test/tracking_results/``` contains raw results for all datasets. These results can be analyzed using the [analysis](lib/test/analysis) module in lib/test. 