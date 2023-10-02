#!/usr/bin/env python3
import numpy as np
import scipy
import scipy.optimize
import os
import sys
import random
import math
import time

mode=int(sys.argv[1])

# atom kernel properties
atom_ktype=1
compare_elem=False

leave_one_out=True

# other properties
max_nodes=300000 # Don't use structures with more than this many nodes
weight_a=0.5 # weight of atom relative to neighbors (used in OA kernel)

# csv file with descriptors
import csv
"""
csvdescs=[]
with open("2019-11-01-ASR-internal_14142.csv",'r') as f:
        csvr=csv.reader(f, delimiter=',')
        for i,line in enumerate(csvr):
                if i == 0:
                        csvheader=line
                else:
                        csvdescs.append(line)
"""

# Read in data about elements
radii=dict()
vdwradii=dict()
anums=dict()
ismetal=dict()
symbols=dict()
with open("elements.txt",'r') as f:
        for line in f:
                anum_,elem,ismetal_,radius_,vdwradius_=line.split()
                anums[elem]=int(anum_)
                ismetal[elem]=int(ismetal_)
                radii[elem]=float(radius_)
                vdwradii[elem]=float(vdwradius_)
                symbols[int(anum_)]=elem

class node: #Represents a node in a graph representation
        z=-1
        val=-1
        sasa=-999
        q=-999
        q_agg=-999
        radius=-999
        potential=-999
        element="None"
        def __init__(self,element="*",z=-1,val=-1,sasa=-999,q=-999,q_agg=-999,potential=-999):
                self.val=val
                self.sasa=sasa
                self.q=q
                self.q_agg=q_agg
                self.potential=potential

                if element in anums:
                        self.element=element
                        self.z=anums[element]
                        self.radius=radii[element]
                elif z > 0:
                        self.element=[v for v in anums if anums[v] == z][0]
                        self.z=z
                        self.radius=radii[self.element]

        def fromstr(string):
                n=node()
                j, n.element, z_, val_, sasa_, q_, q_agg_, radius_, potential_ = string.split()
                n.z=int(z_)
                n.val=int(val_)
                n.sasa=float(sasa_)
                n.q=float(q_)
                n.q_agg=float(q_agg_)
                n.radius=float(radius_)
                n.potential=float(potential_)
                return n

        def __str__(self):
                return "Node\t%s\t%d\t%d\t%f\t%f\t%f\t%f\t%f"%(self.element,self.z, self.val, self.sasa, self.q, self.q_agg, self.radius, self.potential)

# Kernel function between two nodes/atoms ***** for MOFs

def k_atom(a1,a2,params_node):
        #print(a1.q_agg-a2.q_agg,a1.radius-a2.radius)
        if a1.z == a2.z:
                if a1.val == a2.val:
                        return 1.0
                else:
                        return 0.7
        else:
                return 0.3 * math.exp(-.5*((a1.q_agg-a2.q_agg)/params_node[0])**2-.5*((a1.radius-a2.radius)/params_node[1])**2)

"""
def k_atom(a1,a2,params_node):
        if a1.z == a2.z and a1.val == a2.val:
                return 1
        if a1.z == a2.z and a1.val != a2.val:
                return .7
        else:
                return 0
"""
################################## Read Charge Assignment Params. #################
q_2=dict()
q_1=dict()
q_0=dict()
with open("../m-CBAC/Database_2nd.txt",'r') as f:
        for line in f:
                sp=line.split()
                if len(sp)<=1:
                        continue
                key=sp[0]+"_"+sp[1]+"_"+sp[2]
                q=float(sp[3])
                q_2[key]=q
with open("../m-CBAC/Database_1st.txt",'r') as f:
        for line in f:
                sp=line.split()
                if len(sp)<=1:
                        continue
                key=sp[0]+"_"+sp[1]
                q=float(sp[2])
                q_1[key]=q
with open("../m-CBAC/Database_0th.txt",'r') as f:
        for line in f:
                sp=line.split()
                if len(sp)<=1:
                        continue
                key=sp[0]
                q=float(sp[1])
                q_0[key]=q

#########################################################################

class mol:
        a=None
        d=None # distance matrix
        neighs=None # neighbors within some distance thresh
        name=""
        id=-1

        pg=""

        # descriptors
        flexibility=-1
        dipolemag=-1
        maxdist=-1
        cr_vol=-1

        dipole=[0,0,0]

        #def savexyz(self,fname):
        #       with open(fname,'w') as f:
        #               f.write("%d\t\t"%len(self.a))
        #               for at in self.a:
        #                       f.write("%d\t%f\t%f\t%f\n"%(at[0],at[1],at[2],at[3]))

        def savemol(self,path):
                with open(path,'w') as f:
                        f.write("Name\t%s"%self.name)
                        f.write("\nID\t%d"%self.id)
                        f.write("\nFlexibility\t%f"%self.flexibility)
                        f.write("\nPointgroup\t%s"%self.pg)
                        f.write("\nCR_Vol\t%f"%self.cr_vol)
                        f.write("\nMaxDist\t%f"%self.maxdist)
                        f.write("\nDipoleMag\t%f"%self.dipolemag)
                        f.write("\nNodes:")
                        for i,at in enumerate(self.a):
                                f.write("\n%d\t"%i)
                                f.write(str(at))
                        f.write("\nDists:")
                        for i in range(len(self.a)):
                                for j in range(len(self.a)):
                                        if i == j:
                                                continue
                                        f.write("\n%d\t%d\t%f"%(i,j,self.d[i,j]))

        def __init__(self,xyz):
                if xyz[-4:] == ".mol":
                        with open(xyz,'r') as f:
                                self.a=[]
                                sec=-1 # 1 = nodes, 2 = dists
                                for line in f:
                                        if "Name" in line:
                                                self.name=line.split()[1]
                                        elif "ID" in line:
                                                self.id=int(line.split()[1])
                                        elif "Flexibility" in line:
                                                self.flexibility=float(line.split()[1])
                                        elif "Pointgroup" in line:
                                                self.pg=line.split()[1]
                                        elif "CR_Vol" in line:
                                                self.cr_vol=float(line.split()[1])
                                        elif "MaxDist" in line:
                                                self.maxdist=float(line.split()[1])
                                        elif "DipoleMag" in line:
                                                self.dipolemag=float(line.split()[1])
                                        elif "Nodes:" in line:
                                                sec=1
                                        elif "Dists:" in line:
                                                sec=2
                                                self.d=np.zeros((len(self.a),len(self.a)))
                                        elif sec == 1:
                                                nstr=line.split(maxsplit=1)[1]
                                                self.a.append(node.fromstr(nstr))
                                        elif sec == 2:
                                                i_,j_,d_=line.split()
                                                i=int(i_)
                                                j=int(j_)
                                                d=float(d_)
                                                self.d[i,j]=d
                else:
                        if xyz[-4:] == '.xyz':
                                ats=[]
                                with open(xyz,'r') as f:
                                        for i,line in enumerate(f):
                                                if i >= 2:
                                                        sp=line.split()
                                                        if len(sp) >= 4:
                                                                ats.append([sp[0],float(sp[1]),float(sp[2]),float(sp[3]),-1]) # ***** add valence
                        else:
                                """
                                import chemml
                                from chemml.chem import Molecule
                                m = Molecule(xyz,'smiles')
                                m.to_xyz(optimizer='UFF')

                                ats=[]
                                for i,s in enumerate(m.atomic_symbols):
                                        ats.append([s,m.xyz.geometry[i,0],m.xyz.geometry[i,1],m.xyz.geometry[i,2]])
                                """

                                from rdkit import Chem
                                from rdkit.Chem import AllChem
                                m = Chem.MolFromSmiles(xyz)
                                m = Chem.AddHs(m)

                                AllChem.EmbedMultipleConfs(m)
                                engs=AllChem.MMFFOptimizeMoleculeConfs(m)

                                mineng=min(engs)
                                best=engs.index(mineng)

                                self.cr_vol=AllChem.ComputeMolVolume(m,best)**(1.0/3)

                                m_noH=Chem.RemoveAllHs(m)
                                rms=[AllChem.GetBestRMS(m_noH, m_noH, best, i) for i in range(len(m.GetConformers()))]
                                self.flexibility=max(rms)/len(m.GetAtoms())

                                self.atoms=[]
                                #print("nconf",len(m.GetConformers()))
                                c=m.GetConformer(best).GetPositions()
                                for i,a in enumerate(m.GetAtoms()):
                                        self.atoms.append([a.GetAtomicNum(),c[i][0],c[i][1],c[i][2],a.GetDegree()])

                                # test symmetry
                                from pymatgen.core.structure import Molecule
                                pmgmol=Molecule([a[0] for a in self.atoms],[a[1:4] for a in self.atoms])
                                from pymatgen.symmetry.analyzer import PointGroupAnalyzer
                                pga = PointGroupAnalyzer(pmgmol)
                                self.pg=pga.sch_symbol

                                self.bonds=[]
                                for i in range(len(self.atoms)):
                                        self.bonds.append([])
                                for b in m.GetBonds():
                                        i1=b.GetBeginAtomIdx()
                                        i2=b.GetEndAtomIdx()
                                        self.bonds[i1].append(i2)
                                        self.bonds[i1].append(i2)
                        q_list=self.__assigncharges()

                        def dist(i1,i2):
                                return sum([(self.atoms[i1][j+1]-self.atoms[i2][j+1])**2 for j in range(3)]) ** .5

                        # q_agg
                        q_agg=[q for q in q_list]
                        for i1 in range(len(self.atoms)):
                                for i2 in range(i1):

                                        d12=dist(i1,i2)

                                        """
                                        if d12 < 2.5:
                                                q_agg[i1]+=q_list[i2]
                                                q_agg[i2]+=q_list[i1]
                                        """

                                        rad=200.0
                                        w=math.exp(-d12/rad)
                                        q_agg[i1]+=w*q_list[i2]
                                        q_agg[i2]+=w*q_list[i1]

                        # dipole moment
                        self.dipole=[0,0,0]
                        for i,a in enumerate(self.atoms):
                                for j in range(3):
                                        self.dipole[j]+=q_list[i]*self.atoms[i][j+1]
                        self.dipolemag=(self.dipole[0]**2+self.dipole[1]**2+self.dipole[2]**2)**.5

                        if True: # Remove H
                                inds=[i for i in range(len(self.atoms)) if self.atoms[i][0]!=1]
                                q_list=[q_list[i] for i in inds]
                                q_agg=[q_agg[i] for i in inds]
                                self.atoms=[self.atoms[i] for i in inds]

                        self.d=np.zeros((len(self.atoms),len(self.atoms)))
                        self.neighs=[]
                        for i in range(len(self.atoms)):
                                self.neighs.append([])

                        self.maxdist=0
                        for i1,a1 in enumerate(self.atoms):
                                for i2 in range(i1+1):
                                        a2=self.atoms[i2]

                                        d12=dist(i1,i2)
                                        self.maxdist=max(self.maxdist,d12)
                                        self.d[i1,i2]=d12
                                        self.d[i2,i1]=d12
                                        if d12 < 100:
                                                self.neighs[i1].append(i2)
                                                self.neighs[i2].append(i1)


                        self.a=[]
                        for i,a in enumerate(self.atoms):
                                self.a.append(node(z=a[0],val=a[4],q=q_list[i],q_agg=q_agg[i]))

        def __assigncharges(self):
                # Assign charges to atoms using method described in reference below:
                # Efficient and Accurate Charge Assignments via a Multilayer Connectivity-Based Atom Contribution (m-CBAC) Approach
                # https://pubs.acs.org/doi/abs/10.1021/acs.jpcc.0c01524
                q_list=[]
                self.c_2=0
                self.c_1=0
                self.c_0=0
                self.c_n=0
                for i_0,a in enumerate(self.atoms):
                        n_0=symbols[a[0]]
                        n_1=[]
                        n_2=[]
                        for i_1 in self.bonds[i_0]:
                                #print(i_1,len(self.atoms))
                                n_1.append(symbols[self.atoms[i_1][0]])
                                for i_2 in self.bonds[i_1]:
                                        n_2.append(symbols[self.atoms[i_2][0]])

                        n_1=sorted(n_1)
                        n_2=sorted(n_2)

                        key_2=n_0+"_"+"".join(n_1)+"_"+"".join(n_2)
                        key_1=n_0+"_"+"".join(n_1)

                        q=None
                        #print(key_2,n_0)
                        if key_2 in q_2:
                                q=q_2[key_2]
                                #print("key 2")
                                self.c_2+=1
                        elif key_1 in q_1:
                                q=q_1[key_1]
                                #print("key 1")
                                self.c_1+=1
                        elif n_0 in q_0:
                                q=q_0[n_0]
                                #print("key 0")
                                self.c_0+=1
                        else:
                                q=0
                                #print("no key")
                                self.c_n+=1

                        q_list.append(q)

                q_tot=sum(q_list)
                q_shift=q_tot/len(q_list)
                #print("q_tot=%f,\tq_shift=%f"%(q_tot,q_shift))
                q_list=[q-q_shift for q in q_list]
                return q_list

        def get_descs(self):
                return [self.flexibility, self.maxdist, self.cr_vol, self.dipolemag]


#############################################################################
# descriptor kernel function
def k_desc(d1, d2, params_desc):
        if len(d1) != len(params_desc) or len(d2) != len(params_desc):
                print(len(d1),len(d2),len(params_desc))
                raise ValueError
        ssq=0
        for i in range(len(params_desc)):
                ssq += ((d1[i]-d2[i])/params_desc[i])**2
        return math.exp(-ssq/2)

fixneg=True
columnconstraint=True
symmetrize=True
filewrite=False
# graph kernel function (one of the two kernel types defined above
def k_asym(m1,m2,params):
        d_weight=params[0]
        d_kernel=params[1]
        w_d=params[2]
        params_node=params[3:]

        w_ka=(1-w_d)/2
        def V(d):
                return math.exp(-d/d_weight)
                #return 1/(d+d_weight)**3
        def D(d1,d2):
                if d1+d2 == 0:
                        return 1
                r=2*abs(d1-d2)/(d1+d2)
                return math.exp(-r/d_kernel)

        def makeweights(m):
                weights=np.zeros((len(m.a),len(m.a)))
                for i in range(len(m.a)):
                        w=[V(m.d[i,k]) for k in range(len(m.a))]
                        #w=[V(m.d[i,k]) if k != i else 0 for k in range(len(m.a))]
                        wtot=sum(w)
                        for k,f in enumerate(w):
                                weights[i,k]=f/wtot
                return weights

        v1=makeweights(m1)
        v2=makeweights(m2)

        m=len(m1.a)
        n=len(m2.a)

        atomkernels=np.zeros((m,n))
        for i in range(m):
                for j in range(n):
                        atomkernels[i,j]=k_atom(m1.a[i],m2.a[j],params)

        def deindex(p):
                return p%m, int(p/m)
        def index(i,j):
                return j*m + i
        H=np.zeros((m*n,m*n))
        H2=np.zeros((m*n,m*n))
        for p in range(m*n):
                #print("Hp",p,m*n)
                i,j=deindex(p)
                for q in range(p+1): # possibly doubly redundant
                        k,l=deindex(q)
                        dk=D(m1.d[i,k],m2.d[j,l])
                        H[p,q]=.25*((v1[i,k]+v1[k,i])/m+(v2[j,l]+v2[j,l])*n/m**2)*((1-dk)*w_d+(1-atomkernels[i,j])*w_ka+(1-atomkernels[k,l])*w_ka)
                        H[q,p]=H[p,q]
                        th2=(dk*w_d+atomkernels[i,j]*w_ka+atomkernels[k,l]*w_ka)
                        H2[p,q]=.5*(v1[i,k]/m+v2[j,l]*n/m**2)*th2
                        H2[q,p]=.5*(v1[k,i]/m+v2[l,j]*n/m**2)*th2
        if filewrite:
                with open("H.txt",'w') as f:
                        f.write(str(m*n))
                        for j in range(m*n):
                                f.write("\t%d"%j)
                        for i in range(m*n):
                                f.write("\n%d"%i)
                                for j in range(m*n):
                                        f.write("\t%f"%(H[i,j]))

        from cvxopt import matrix, solvers
        G=[]
        J=[]
        for i in range(m*n):
                G.append([0.0]*m*n)
                G[-1][i]=-1.0
                J.append(0.0)

        A=[]
        B=[]
        for i in range(m):
                A.append([0.0]*m*n)
                B.append(1.0)
                for j in range(n):
                        A[-1][index(i,j)]=1.0
        if columnconstraint:
                for j in range(1,n):
                        A.append([0.0]*m*n)
                        B.append(0.0)
                        for i in range(m):
                                A[-1][index(i,j)]=1
                                A[-1][index(i,0)]=-1

        Q=[0.0]*m*n
        eigs,evecs=np.linalg.eig(H)
        mineig=min(eigs)
        print("mineig",mineig)
        if mineig < 0 and fixneg:
                for i in range(m*n):
                        if eigs[i] < 0:
                                eigs[i]=0
                D=np.diag(eigs)
                H=np.dot(evecs,np.dot(D,np.linalg.inv(evecs)))

        if filewrite:
                with open("eigs.txt",'w') as f:
                        for eig in eigs:
                                f.write("%f\n"%eig)

                with open("Hs.txt",'w') as f:
                        f.write(str(m*n))
                        for j in range(m*n):
                                f.write("\t%d"%j)
                        for i in range(m*n):
                                f.write("\n%d"%i)
                                for j in range(m*n):
                                        f.write("\t%f"%(H[i,j]))

        sol = solvers.qp(matrix(np.real(H)),matrix(Q),matrix(G).T,matrix(J),matrix(A).T,matrix(B))
        if sol['status'] == 'optimal':
                q=sol['x']

                with open("q.txt",'w') as f:
                        f.write(str(m))
                        for j in range(n):
                                f.write("\t%d"%j)
                        for i in range(m):
                                f.write("\n%d"%i)
                                for j in range(n):
                                        f.write("\t%f"%(q[index(i,j)]))

        res=np.dot(q.T,np.dot(H2,q))
        return res[0,0]



def k_3D(m1,m2,params):
        k12=k_asym(m1,m2,params)
        return k12

def readdata(dataset):
        mols=[]
        with open("data.txt",'r') as f:
                for line in f:
                        id_,smiles,name,y_,j,cl_,s=line.split()
                        id=int(id_)
                        if s != dataset:
                                continue
                        print(name)
                        try:
                                #m=mol(smiles) # actually xyz path, not smiles
                                m=mol("mols/m_%d.mol"%id)
                                if len(m.a) > 35:
                                        print("rejected for size")
                                        continue
                                if m.flexibility > 0.05:
                                        print("rejected for flexibility")
                                        continue
                                if m.pg != "C1":
                                        print("rejected for symmetry (%s)"%m.pg)
                                        continue
                                m.y=float(y_)
                                m.id=int(id_)
                                m.name=name
                                m.cl = 1 if "+" in cl_ else 0
                                mols.append(m)
                        except:
                                print("failed")
        print("%d mols read"%(len(mols)))
        return mols

def write_predictor(fname,mofs,mean,params,params_outer,obj,predictor,Kg_self,invgram=None):
        # Save results:
        with open(fname,'w') as f:
                f.write("objective\t%f\n"%obj)
                f.write("params\t%s\n"%("\t".join([str(v) for v in params])))
                f.write("params_outer\t%s\n"%("\t".join([str(v) for v in params_outer])))
                f.write("mean\t%f\n"%mean)
                f.write("data:\n")
                for i,m in enumerate(mofs):
                        f.write("%d\t%d\t%f\t%f\n"%(i,m.id,predictor[i],Kg_self[i]))
                if invgram is not None:
                        f.write("invgram:\n")
                        for i in range(len(mofs)):
                                f.write("%d\t"%i)
                                f.write("\t".join([str(invgram[i,j]) for j in range(i+1)]))
                                f.write("\n")

def read_predictor(fname):
        mean=None
        params=None
        outerparams=None
        idlist=[]
        selfkernel=[]
        predictor=[]
        invgram=None
        with open(fname,'r') as f:
                section=-1
                for i,line in enumerate(f):
                        if "params_outer" in line:
                                outerparams=[float(v) for v in line.split()[1:]]
                        elif "params" in line:
                                params=[float(v) for v in line.split()[1:]]
                        elif "mean" in line:
                                mean=float(line.split()[1])
                        elif "data:" in line:
                                section=1
                        elif "invgram:" in line:
                                section=2
                                invgram=np.zeros((len(idlist),len(idlist)))
                        else:
                                if section == 1:
                                        sp=line.split()
                                        if len(sp) >= 3:
                                                idlist.append(int(sp[1]))
                                                selfkernel.append(float(sp[3]))
                                                predictor.append(float(sp[2]))
                                elif section == 2:
                                        sp=line.split()
                                        i=int(sp[0])
                                        for j,v_ in enumerate(sp[1:]):
                                                v=float(v_)
                                                invgram[i,j]=v
                                                invgram[j,i]=v

        return mean,params,outerparams,idlist,selfkernel,predictor,invgram

def predict(prednum,valset):
        mean,params,outerparams,idlist,K_self,predictor,invgram=read_predictor("trainingstuff/predictor_%d.txt"%prednum)
        a,b,c,w=params[:4]
        params_desc=params[4:]

        mols=[]
        for id in idlist:
                mols.append(mol("mols/m_%d.mol"%id))

        fname="trainingstuff/prediction_%d.txt"%prednum
        for nv,mv in enumerate(valset):
                k_mv_mv=k_3D(mv,mv,outerparams)
                pred=mean
                if invgram is not None:
                        ks=[]
                for i,mt in enumerate(mols):
                        k_mv_mt=k_3D(mv,mt,outerparams)
                        k_mv_mt/=(k_mv_mv*K_self[i])**.5
                        kd=k_desc(mv.get_descs(),mt.get_descs(),params_desc)

                        k=a*(w*k_mv_mt+(1-w)*kd)**c
                        pred+=k*predictor[i]
                        if invgram is not None:
                                ks.append(k)
                        print("validation # ", nv, i,"/",len(mols))
                        #f.write("%d\t%d\t%f\t%f\t%f\t%f\n"%(mv.id,mt.id,k_mv_mt,kd,k,k*predictor[i]))
                if invgram is None:
                        k_self=a*(w*k_mv_mv+1-w)**c
                        var=k_self-np.dot(ks,np.dot(invgram,ks))
                else:
                        var=-999999
                print(mv.name,pred,mv.y,k_mv_mv,var)
                os.system("echo \"%s\t%f\t%f\t%f\t%f\" >> %s"%(mv.name,pred,mv.y,k_mv_mv,var,fname))

def predict_svm(prednum,valset):
        import pickle
        with open("trainingstuff/model_%d.svm"%prednum,'rb') as f:
                clf=pickle.load(f)

        """
        mols=[]
        for id in idlist:
                mols.append(mol("mols/m_%d.mol"%id))
        """

        fname="trainingstuff/predictionsvm_%d.txt"%prednum
        for nv,mv in enumerate(valset):
                ys=clf.predict(mv)
                print(mv.name,ys,mv.cl,fname)
                os.system("echo \"%d\t%d\t%d\" >> %s"%(mv.id,ys,mv.cl,fname))

class svm_wrapper:
        clf=None
        #mol_ids=[]
        K=None
        K_diag=None
        outerparams=None
        params_desc=None
        a=None
        b=None
        c=None
        w=None
        mols=[]
        def __init__(self,mols,K,K_diag,outerparams,innerparams):
                from sklearn import svm
                #mol_ids=[m.id for m in mols]
                self.mols=mols
                self.K=np.copy(K)
                self.K_diag=np.copy(K_diag)
                self.outerparams=outerparams
                self.a, self.b, self.c, self.w = innerparams[:4]
                self.params_desc=innerparams[4:]
                self.clf=svm.SVC(kernel='precomputed',class_weight='balanced')
                print([m.cl for m in mols])
                self.clf.fit(K,[m.cl for m in mols])
        def predict(self, x):
                k_xx=k_3D(x,x,self.outerparams)
                ks=[]
                for i,m in enumerate(self.mols):
                        k_xm=k_3D(x,m,self.outerparams)
                        k_xm/=(k_xx*self.K_diag[i])**.5
                        kd=k_desc(x.get_descs(),m.get_descs(),self.params_desc)

                        k=self.a*(self.w*k_xm+(1-self.w)*kd)**self.c
                        ks.append(k)

                return self.clf.predict([ks])


if mode == 0:
        # convert molecs to custom format
        os.system("mkdir mols")
        with open("data.txt",'r') as f:
                for line in f:
                        id_,smiles,name,y_,j,j,s=line.split()
                        print(name)
                        #m=mol(smiles)
                        #print(1/0)
                        try:
                                m=mol(smiles)
                                m.y=float(y_)
                                m.id=int(id_)
                                m.name=name
                                m.savemol("mols/m_%d.mol"%(m.id))

                        except:
                                print("failed")

elif mode == 1:
        # just print out the training data
        allmols=readdata("Training")
        y=[m.y for m in allmols]
        print("mean",sum(y)/len(y))
        print("#",len(allmols))
        print([m.flexibility for m in allmols])

elif mode == 2:
        # validation
        prednum=int(sys.argv[2])
        validation=readdata("Test")
        predict(prednum,validation)

elif mode == 3:
        # args: <mode> <predictor number> <group> <svm?>
        #paralell validation group, used by mode 4
        prednum=int(sys.argv[2])
        group=int(sys.argv[3])
        use_svm=int(sys.argv[4])
        valset=[]
        with open("trainingstuff/groupids_%d.txt"%group,'r') as f:
                for line in f:
                        sp=line.split()
                        if len(sp)> 1:
                                id=int(sp[0])
                                m=mol("mols/m_%d.mol"%id)
                                if use_svm:
                                        m.cl=int(sp[1])
                                else:
                                        m.y=float(sp[1])
                                valset.append(m)

        if use_svm:
                predict_svm(prednum,valset)
        else:
                predict(prednum,valset)
        os.system("rm trainingstuff/groupids_%d.txt"%group)

elif mode == 4:
        # args: <mode> <predictor number> <svm?> <# groups>
        # validate in paralell
        prednum=int(sys.argv[2])
        n_groups=int(sys.argv[4])
        use_svm=int(sys.argv[3])
        validation=readdata("Test")
        #validation=validation[:5]+validation[-5:]

        for group in range(n_groups):
                gi=[m for j,m in enumerate(validation) if j%n_groups == group]
                with open("trainingstuff/groupids_%d.txt"%group,'w') as f:
                        for m in gi:
                                if use_svm:
                                        f.write("%d\t%d\n"%(m.id,m.cl))
                                else:
                                        f.write("%d\t%f\n"%(m.id,m.y))
                os.system("cp val_n.sh trainingstuff/val_%d.sh"%group)
                os.system("sed -i \"s/<n>/%d/\" trainingstuff/val_%d.sh"%(group,group))
                os.system("sed -i \"s/<p>/%d/\" trainingstuff/val_%d.sh"%(prednum,group))
                os.system("sed -i \"s/<v>/%d/\" trainingstuff/val_%d.sh"%(use_svm,group))
                os.system("sbatch trainingstuff/val_%d.sh"%group)

elif mode == 5:
        # paralell kernel evaluations, used by mode 6. Args: <mode> <index>
        # Given the list of pairs to evaluate in trainingstuff/pairs_<index>.txt, output the kernel values to trainingstuff/pairs_<index>.k, then create trainingstuff/pairs_<index>.fin when finished
        ind=int(sys.argv[2])

        with open("trainingstuff/params.txt",'r') as f:
                params=[float(v) for v in f.read().split()]

        mols=[]
        with open("trainingstuff/mollist.txt",'r') as f:
                for line in f:
                        if len(line) < 2:
                                continue
                        id=int(line)
                        mols.append(mol("mols/m_%d.mol"%id))
        with open("trainingstuff/pairs_%d.k"%ind,'w') as out:
                with open("trainingstuff/pairs_%d.txt"%ind,'r') as f:
                        for line in f:
                                sp=line.split()
                                if len(sp) != 2:
                                        continue
                                i1,i2=[int(v) for v in sp]
                                k=k_asym(mols[i1],mols[i2],params)
                                out.write("%d\t%d\t%f\n"%(i1,i2,k))
        os.system("touch trainingstuff/pairs_%d.fin"%ind)

elif mode == 6:
        use_svm=int(sys.argv[2])
        n_groups=32

        mols=readdata("Training")
        #mols=mols[:5]+mols[-5:] # ***** delete
        with open("trainingstuff/mollist.txt",'w') as f:
                for m in mols:
                        f.write("%d\n"%m.id)

        y=[m.cl if use_svm else m.y for m in mols]

        nmols=len(mols)
        pairs=[]
        for i in range(nmols):
                for j in range(i+1):
                        pairs.append((i,j))
        import random
        random.seed(123)
        random.shuffle(pairs)

        pairs_groups=[]
        for n in range(n_groups):
                pairs_groups.append([])
        for i,pair in enumerate(pairs):
                pairs_groups[i%n_groups].append(pair)

        for n,pl in enumerate(pairs_groups):
                with open("trainingstuff/pairs_%d.txt"%n,'w') as f:
                        for i1,i2 in pl:
                                f.write("%d\t%d\n"%(i1,i2))
                os.system("cp run_n.sh trainingstuff/run_%d.sh"%n)
                os.system("sed -i \"s/<n>/%d/\" trainingstuff/run_%d.sh"%(n,n))


        def tos_params(params):
                return "\t".join([str(v) for v in params])

        it=0
        def objective(params):
                global it
                global innerparams

                def isfinished():
                        for n in range(n_groups):
                                if not os.path.exists("trainingstuff/pairs_%d.fin"%n):
                                        return False
                        return True
                def clearjunk():
                        os.system("rm trainingstuff/pairs_*.fin")
                        os.system("rm trainingstuff/pairs_*.k")
                        with open("trainingstuff/params.txt",'w') as f:
                                f.write("\t".join([str(v) for v in params]))

                def runall():
                        for n in range(n_groups):
                                os.system("sbatch --output=/dev/null trainingstuff/run_%d.sh"%n)

                clearjunk()
                runall()

                while not isfinished():
                        time.sleep(10)

                os.system("cat trainingstuff/pairs_*.k > trainingstuff/allpairs_%d.txt"%it)

                K=np.identity(nmols)
                K_diag=[-1]*nmols
                with open("trainingstuff/allpairs_%d.txt"%it,'r') as f:
                        for line in f:
                                sp=line.split()
                                if len(sp) >= 3:
                                        i=int(sp[0])
                                        j=int(sp[1])
                                        k=float(sp[2])**3
                                        if i == j:
                                                K_diag[i]=k
                                        else:
                                                K[i,j]=k
                                                K[j,i]=k


                # remove points with low self-kernel. Make negative for no removal.
                thresh=-.9
                i_del=[i for i in range(nmols) if K_diag[i] < thresh]
                i_keep=[i for i in range(nmols) if i not in i_del]
                mols_sub=[mols[i] for i in i_keep]
                nmols_sub=len(i_keep)
                K_diag_sub=[K_diag[i] for i in i_keep]
                y_sub=[y[i] for i in i_keep]
                K_sub=np.delete(np.delete(K,i_del,0),i_del,1)
                print("ikeep",i_keep)
                print("idel", i_del)

                if not use_svm:
                        mean_sub=sum(y)/len(y)
                        y_sub=[v-mean_sub for v in y_sub]
                        print("mean",mean_sub)
                # normalize
                for i in range(nmols_sub):
                        K_sub[i,i]=1
                        for j in range(i):
                                Kij=K_sub[i,j]/(K_diag_sub[i]*K_diag_sub[j])**.5
                                K_sub[i,j]=Kij
                                K_sub[j,i]=Kij


                # PSD check
                mineig=min(np.linalg.eigvals(K_sub))
                print("minimum eigenvalue!!!",mineig)


                # PSD enforcement
                while mineig < 0:
                        i_best=-1
                        best=mineig
                        for i in range(nmols_sub):
                                K_i=np.delete(np.delete(K_sub,i,0),i,1)
                                me=min(np.linalg.eigvals(K_i))
                                if me > best:
                                        best=me
                                        i_best=i
                        print("minimum eigenvalue",best)

                        print("deleting %d, %s"%(mols_sub[i_best].id, mols_sub[i_best].name))
                        nmols_sub -= 1
                        K_sub=np.delete(np.delete(K_sub,i_best,0),i_best,1)
                        del K_diag_sub[i_best]
                        del y_sub[i_best]
                        del mols_sub[i_best]
                        mineig=best

                def objective_inner(iparams,savepredictor):
                        a,b,c,w=iparams[:4]
                        params_desc=iparams[4:]
                        if b < 0 or w < 0 or w > 1:
                                return float("NaN")

                        K2=np.zeros((nmols_sub,nmols_sub))
                        for i in range(nmols_sub):
                                for j in range(nmols_sub):
                                        K_desc=k_desc(mols_sub[i].get_descs(),mols_sub[j].get_descs(),params_desc)
                                        K2[i,j]=a*(w*K_sub[i,j]+(1-w)*K_desc)**c

                        mineig=min(np.linalg.eigvals(K2))+b
                        if mineig < 0:
                                print("Not PSD",mineig, a, b, c, w)
                                return float("NaN")

                        if use_svm:
                                clf=svm_wrapper(mols_sub,K2,K_diag_sub,params,iparams)
                                pred=clf.clf.predict(K2)
                                #obj=sum([1 if pred[i] == y_sub[i] else 0 for i in range(len(y_sub))])/float(len(y_sub))
                                n11=sum([1 if pred[i] == 1 and y_sub[i] == 1 else 0 for i in range(len(y_sub))])
                                n01=sum([1 if pred[i] == 0 and y_sub[i] == 1 else 0 for i in range(len(y_sub))])
                                n00=sum([1 if pred[i] == 0 and y_sub[i] == 0 else 0 for i in range(len(y_sub))])
                                n10=sum([1 if pred[i] == 1 and y_sub[i] == 0 else 0 for i in range(len(y_sub))])
                                obj1=float(n11)/(n11+n01)
                                obj0=float(n00)/(n00+n10)
                                obj=(obj1+obj0)/2
                                if savepredictor >= 0:
                                        import pickle
                                        with open("trainingstuff/model_%d.svm"%savepredictor,'wb') as f:
                                                pickle.dump(clf,f)
                                print("obj",obj)
                                return obj
                        else:
                                K2_b=K2+b*np.identity(nmols_sub)
                                K2_b_inv=np.linalg.inv(K2_b)

                                res=0
                                for i in range(nmols_sub):
                                        #K2_i=np.delete(np.delete(K2,i,0),i,1)
                                        # calculate submatrix inv,
                                        v=K2_b[i,:]
                                        P=np.zeros((nmols_sub,2))
                                        Q=np.zeros((2,nmols_sub))
                                        for j in range(nmols_sub):
                                                if j == i:
                                                        P[j,0]=1
                                                        Q[1,j]=1
                                                else:
                                                        P[j,1]=K2_b[j,i]
                                                        Q[0,j]=K2_b[j,i]

                                        t=np.linalg.inv(np.dot(Q,np.dot(K2_b_inv,P))-np.identity(2))
                                        tt=K2_b_inv-np.dot(K2_b_inv,np.dot(P,np.dot(t,np.dot(Q,K2_b_inv))))
                                        K2_b_inv_i=np.delete(np.delete(tt,i,0),i,1)


                                        y_i=[y_sub[j] for j in range(nmols_sub) if j != i]
                                        K2s_i=np.delete(K2,i,1)[i,:]

                                        ys=np.dot(K2s_i,np.dot(K2_b_inv_i,y_i))

                                        ys_var=K2[i,i]-np.dot(K2s_i,np.dot(K2_b_inv_i,K2s_i))
                                        res -= (ys-y[i])**2/(2*ys_var)+math.log(ys_var)/2
                                        print(i,ys,ys_var,res)

                                obj=res/nmols_sub

                                if savepredictor >= 0:
                                        K2+=b*np.identity(nmols_sub)
                                        predictor=np.dot(np.linalg.inv(K2),y_sub)
                                        write_predictor("trainingstuff/predictor_%s.txt"%savepredictor,mols_sub,mean_sub,iparams,params,obj,predictor,K_diag_sub,K2_b_inv)

                                print(obj)
                                return obj

                opt_inner=scipy.optimize.minimize(lambda iparams: -objective_inner(iparams,-1),[.1,.01,1.0,.5]+[0.05,5,5,.2],method="nelder-mead",options={'xtol': 1e-5, 'disp': True})
                obj=objective_inner(opt_inner.x,it)
                with open("trainingstuff/allparams.txt",'a') as f:
                        f.write("%d;\t%f;\t%s;\t%s\n"%(it,obj,tos_params(params),tos_params(opt_inner.x)))

                it += 1
                return obj

        opt_outer=scipy.optimize.minimize(lambda params: -objective(params),[10.0, 0.25, 0.3 , 0.3, 0.6],method="nelder-mead",options={'xtol': 1e-5, 'disp': True})

        with open("trainingstuff/allparams.txt",'a') as f:
                f.write("final;\t%s;\t%s"%(tos_params(params),tos_params(innerparams)))
