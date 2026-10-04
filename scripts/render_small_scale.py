"""Render measured motion arrays; reference-only mode requires no GPU run."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/small_scale_8gb"
CHAINS = [[0,2,5,8,11],[0,1,4,7,10],[0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]]


def animate(sequences, titles, output):
    sequences = [a[...,:66].reshape(a.shape[0],-1,22,3) for a in sequences]
    points = np.concatenate([a.reshape(-1,3) for a in sequences])
    assert np.isfinite(points).all()
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low+high)/2
    span = max(float((high-low).max())*1.15,2.0)
    fig = plt.figure(figsize=(5*len(sequences),5),facecolor="#f6f8fb")
    axes, lines = [], []
    for i, (array,title) in enumerate(zip(sequences,titles)):
        ax = fig.add_subplot(1,len(sequences),i+1,projection="3d")
        ax.set_title(title,fontsize=12,pad=14)
        ax.set_xlim(center[0]-span/2,center[0]+span/2)
        ax.set_ylim(center[2]-span/2,center[2]+span/2)
        ax.set_zlim(center[1]-span/2,center[1]+span/2)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=15,azim=-60)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_zlabel("Height (m)")
        people = []
        for person in range(array.shape[1]):
            color = ["#2067a4","#df6b24"][person%2]
            people.append([ax.plot([],[],[],color=color,lw=2.3,marker="o",markersize=3)[0] for _ in CHAINS])
        axes.append(ax)
        lines.append(people)
    frame_count = max(a.shape[0] for a in sequences)
    frames = list(range(0,frame_count,2))
    def update(frame):
        for array, people in zip(sequences,lines):
            pose = array[min(frame,len(array)-1)]
            for person, person_lines in enumerate(people):
                for chain,line in zip(CHAINS,person_lines):
                    xyz=pose[person,chain]
                    line.set_data_3d(xyz[:,0],xyz[:,2],xyz[:,1])
        return [line for people in lines for person_lines in people for line in person_lines]
    fig.suptitle("Same spatial scale | No smoothing or per-frame recentering",fontsize=11)
    fig.subplots_adjust(left=.02,right=.98,bottom=.05,top=.85,wspace=.08)
    update(frames[len(frames)//2])
    fig.savefig(output.with_suffix(".png"),dpi=130)
    animation=FuncAnimation(fig,update,frames=frames,interval=1000/15,blit=False)
    animation.save(output,writer=PillowWriter(fps=15))
    plt.close(fig)


def plots(stage):
    rows=[json.loads(line) for line in (OUT/f"{stage}_progress.jsonl").read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout="constrained")
    step=np.array([r["step"] for r in rows])
    loss=np.array([r["loss"] for r in rows])
    axes[0].plot(step,loss,alpha=.35,label="Per update (different t / sample)")
    if len(loss)>=10:
        axes[0].plot(step[9:],np.convolve(loss,np.ones(10)/10,mode="valid"),label="10-update mean")
    axes[0].set(xlabel="Optimizer update",ylabel="Original total loss",title=f"{stage}: optimization diagnostic")
    axes[0].legend(fontsize=8)
    axes[1].plot(step,[r["allocated_mib"] for r in rows],label="Torch allocated")
    axes[1].plot(step,[r["reserved_mib"] for r in rows],label="Torch reserved")
    axes[1].axhline(8151,color="red",ls="--",label="Physical GPU capacity")
    axes[1].set(xlabel="Optimizer update",ylabel="MiB",title="Does not include other applications / driver")
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/f"{stage}_training.png",dpi=150)
    plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",choices=["single","multi"],default="single")
    p.add_argument("--reference-only",action="store_true")
    p.add_argument("--output-dir",type=Path,default=OUT)
    args=p.parse_args()
    OUT=args.output_dir.resolve()
    if args.reference_only:
        data=np.load(OUT/"subset.npz")
        seq=[]
        for split in ["train","val"]:
            length=int(data[f"single_{split}_lengths"][0])
            seq.append(data[f"single_{split}_motions"][0,:length,None,:])
        animate(seq,["Training reference (real)","Validation reference (real)"],OUT/"reference_preview.gif")
        print("Rendered real-data reference only; no generated sample claimed")
    else:
        for split in ["train","val"]:
            before=np.load(OUT/f"{args.stage}_{split}_before.npz")
            after=np.load(OUT/f"{args.stage}_{split}_after.npz")
            assert np.array_equal(before["ground_truth"],after["ground_truth"])
            animate([after["ground_truth"],before["generated"],after["generated"]],
                [f"{split}: real reference","Before trial","After short trial"],OUT/f"{args.stage}_{split}_comparison.gif")
        plots(args.stage)
