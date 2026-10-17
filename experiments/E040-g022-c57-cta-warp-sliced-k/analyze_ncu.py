#!/usr/bin/env python3
"""Export selected NCU report rows and derive cycle-normalized rates."""
import csv, glob, json, os, subprocess
from pathlib import Path
ROOT=Path('/mnt/data/pc01/sglang-v100-migration/experiments/E040')
EXPORT=ROOT/'export'; EXPORT.mkdir(exist_ok=True)
metrics=[
 'gpu__time_duration.avg','smsp__cycles_active.sum','smsp__cycles_elapsed.sum',
 'smsp__warps_eligible.avg','smsp__warps_active.avg','smsp__issue_active.avg',
 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
 'smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct',
 'smsp__warp_issue_stalled_not_selected_per_warp_active.pct',
 'smsp__sass_data_bytes_mem_local.sum','smsp__sass_inst_executed_op_local.sum',
 'smsp__sass_inst_executed_op_global_ld.sum','smsp__sass_inst_executed_op_global_st.sum',
 'dram__bytes_read.sum','dram__bytes_write.sum',
 'lts__t_sectors_aperture_device_op_read.sum','lts__t_sectors_aperture_device_op_write.sum',
 'launch__grid_size','launch__block_size','launch__registers_per_thread',
 'launch__occupancy_limit_registers','launch__occupancy_per_register_count',
 'launch__waves_per_multiprocessor','launch__shared_mem_per_block','launch__stack_size',
]
rows=[]
for rep in sorted((ROOT/'reports').glob('c57-*-20261017T2024*.ncu-rep')):
 out=EXPORT/(rep.stem+'.csv')
 subprocess.run(['/usr/local/cuda/bin/ncu','--import',str(rep),'--csv','--page','raw','--log-file',str(out)],stdout=subprocess.DEVNULL,check=True)
 data=list(csv.reader(out.open())); header=data[0]; values=data[2]; d=dict(zip(header,values))
 cell={'report':rep.name,'report_sha256':subprocess.check_output(['sha256sum',str(rep)],text=True).split()[0]}
 for key in ['Kernel Name','Block Size','Grid Size','gpu__time_duration.avg','smsp__cycles_active.sum','smsp__cycles_elapsed.sum','smsp__warps_eligible.avg','smsp__warps_active.avg','smsp__issue_active.avg','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct','smsp__warp_issue_stalled_not_selected_per_warp_active.pct','smsp__sass_data_bytes_mem_local.sum','smsp__sass_inst_executed_op_local.sum','smsp__sass_inst_executed_op_global_ld.sum','smsp__sass_inst_executed_op_global_st.sum','dram__bytes_read.sum','dram__bytes_write.sum','lts__t_sectors_aperture_device_op_read.sum','lts__t_sectors_aperture_device_op_write.sum','launch__registers_per_thread','launch__occupancy_limit_registers','launch__occupancy_per_register_count','launch__waves_per_multiprocessor','launch__stack_size']:
  cell[key]=d.get(key)
 active=float(cell['smsp__cycles_active.sum']); elapsed=float(cell['smsp__cycles_elapsed.sum'])
 cell['eligible_warps_per_active_cycle']=float(cell['smsp__warps_eligible.avg'])/active
 cell['resident_warps_per_active_cycle']=float(cell['smsp__warps_active.avg'])/active
 cell['issued_warp_inst_per_active_cycle']=float(cell['smsp__issue_active.avg'])/active
 cell['active_cycle_fraction']=active/elapsed
 rows.append(cell)
json_path=EXPORT/'ncu-summary.json'; json_path.write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
for c in rows:
 print(c['report'], 'duration_us='+str(c['gpu__time_duration.avg']), 'eligible/activecycle=%.4f'%c['eligible_warps_per_active_cycle'], 'resident/activecycle=%.4f'%c['resident_warps_per_active_cycle'],'issue/activecycle=%.4f'%c['issued_warp_inst_per_active_cycle'],'active_fraction=%.4f'%c['active_cycle_fraction'],'long='+str(c['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']),'DRAMread='+str(c['dram__bytes_read.sum']))
