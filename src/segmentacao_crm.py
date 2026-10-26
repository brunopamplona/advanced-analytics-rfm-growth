# Segmentação comportamental para CRM - base sintética Claro tv+ (24 meses)
import numpy as np, pandas as pd, pickle, warnings, os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
warnings.filterwarnings('ignore')
# ── Paths (ajuste conforme o ambiente) ─────────────────────────────────────────
CSV     = '/mnt/user-data/outputs/base_assinaturas_clarotv_24m.csv.gz'
OUT_CSV = '/mnt/user-data/outputs/clientes_ativos_segmentados.csv.gz'
PKL_DIR = '/home/claude/crm'; os.makedirs(PKL_DIR, exist_ok=True)
BASE=pd.Timestamp('2024-11-01'); SNAP=pd.Timestamp('2026-10-24'); T0=SNAP-pd.Timedelta(days=90)
cols=['id_cliente','origem','plano','grupo_canal','canal_aquisicao','data_assinatura','data_cancelamento','status','meses_cobrados','preco_lista_mensal','receita_total_paga','receita_tvod',
      'tvod_alugueis','tvod_compras','n_pedidos','cupom_entrada','cupons_tvod_usados','presente','nps','R_dias_sem_acesso','tipo_cancelamento']
df=pd.read_csv(CSV,usecols=cols,parse_dates=['data_assinatura','data_cancelamento'])
for c in ['origem','plano','grupo_canal','canal_aquisicao','status','tipo_cancelamento']: df[c]=df[c].astype('category')
df['canc']=(df.status=='Cancelado').astype('int8')
# ---------- 1) EXPLORACAO (toda a base, 1,92 mi de assinaturas)
expl={}
expl['pedidos']=df.n_pedidos.describe(percentiles=[.1,.25,.5,.75,.9,.99]).round(2)
expl['receita_total']=df.receita_total_paga.describe(percentiles=[.1,.25,.5,.75,.9,.99]).round(2)
expl['recencia_dias']=df.R_dias_sem_acesso.describe(percentiles=[.1,.25,.5,.75,.9,.99]).round(2)
df['cupons_total']=df.cupom_entrada+df.cupons_tvod_usados
expl['cupons_total']=df.cupons_total.describe(percentiles=[.5,.75,.9,.99]).round(2)
expl['pct_com_cupom']=pd.Series({'com cupom de entrada':df.cupom_entrada.mean(),'usou cupom em aluguel':(df.cupons_tvod_usados>0).mean(),'presente':df.presente.mean(),'cancelado':df.canc.mean()})
expl['nps']=df.nps.describe().round(2)
# ---------- RFM + clusters: base ativa em T0 (26/07/2026) com churn observado nos 90 dias seguintes
df['start_d']=(df.data_assinatura-BASE).dt.days; T0_D=(T0-BASE).days; SNAP_D=(SNAP-BASE).days
df['canc_d']=(df.data_cancelamento-BASE).dt.days
p=df[(df.start_d<=T0_D)&((df.status=='Ativo')|(df.canc_d>T0_D))].copy()
# FIX 1: replace(0,1) evita divisão por zero quando meses_cobrados=0 (frac seria inf → clipado a 1 de qualquer forma)
ten=((T0_D-p.start_d)//30+1).astype(float); frac=np.minimum(1,ten/p.meses_cobrados.replace(0,1))
p['R']=p.R_dias_sem_acesso.astype(float)
p['F']=ten+(p.tvod_alugueis+p.tvod_compras)*frac                     # pedidos até T0 (ciclos + aluguéis + compras)
p['M']=ten*p.preco_lista_mensal-np.minimum(ten,2)*p.preco_lista_mensal*0.4*p.cupom_entrada+p.receita_tvod*frac   # valor pago até T0
p['cupom']=p.cupom_entrada+p.cupons_tvod_usados*frac
p['churn90']=((p.status=='Cancelado')&(p.canc_d>T0_D)).astype('int8')
FEATS=['R','F','M','cupom','presente','nps']
def prep(d):
    X=pd.DataFrame({'R':np.log1p(d.R),'F':np.log1p(d.F),'M':np.log1p(d.M),'cupom':np.log1p(d.cupom),'presente':d.presente.astype(float),'nps':d.nps.astype(float)})
    return X
X=prep(p); sc=StandardScaler().fit(X); Z=sc.transform(X).astype('float32')
rng=np.random.default_rng(42); idx=rng.choice(len(Z),150000,replace=False); Zs=Z[idx]
sub=rng.choice(len(Zs),20000,replace=False)
sel=[]
for k in range(3,8):
    km=KMeans(k,n_init=10,random_state=42).fit(Zs); lab=km.labels_
    sel.append(dict(k=k,inercia=km.inertia_,silhouette=silhouette_score(Zs[sub],lab[sub]),calinski_harabasz=calinski_harabasz_score(Zs,lab),davies_bouldin=davies_bouldin_score(Zs,lab),menor_grupo_pct=np.bincount(lab).min()/len(lab)))
sel=pd.DataFrame(sel)
# estabilidade: 5 sementes x amostras diferentes; ARI entre rotulagens aplicadas à mesma amostra de avaliação
# FIX 2: renomeado ev → ev_stab para não colidir com ev da seção 3
ev_stab=rng.choice(len(Z),100000,replace=False); stab=[]
for k in range(3,8):
    labs=[]
    for s in range(5):
        ids=np.random.default_rng(100+s).choice(len(Z),100000,replace=False)
        km=KMeans(k,n_init=5,random_state=s).fit(Z[ids]); labs.append(km.predict(Z[ev_stab]))
    aris=[adjusted_rand_score(labs[i],labs[j]) for i in range(5) for j in range(i+1,5)]
    stab.append(dict(k=k,ARI_medio=np.mean(aris),ARI_min=np.min(aris)))
stab=pd.DataFrame(stab)
# vizinhos: quanto o agrupamento de k se mistura com k+1 (mesma semente)
nest=[]
labk={k:KMeans(k,n_init=10,random_state=42).fit(Zs).predict(Z[ev_stab]) for k in range(3,8)}
for k in range(3,7): nest.append(dict(k=k,ARI_k_vs_k_mais_1=adjusted_rand_score(labk[k],labk[k+1])))
nest=pd.DataFrame(nest)
pickle.dump(dict(expl=expl,sel=sel,stab=stab,nest=nest),open(os.path.join(PKL_DIR,'stage1.pkl'),'wb'))
print(sel.round(3).to_string()); print(stab.round(3).to_string()); print(nest.round(3).to_string())
for k,v in expl.items(): print(k); print(v.to_string() if hasattr(v,'to_string') else v)
print(len(p),p.churn90.mean())


# ---------- 2) ESCOLHA DE K E MODELO FINAL
# k=6: estabilidade equivalente a k=5 (ARI ~0,95), DB menor que k=5, e separa "esfriando com valor" de "quase perdidos" (acoes diferentes)
K=6
km=KMeans(K,n_init=10,random_state=42).fit(Zs); p['g']=km.predict(Z)
def nomear(d,g):
    pr=d.groupby(g).agg(R=('R','median'),F=('F','median'),cupom=('cupom','mean'),presente=('presente','mean'))
    nm={}; pres=pr.presente.idxmax(); nm[pres]='Presenteados'; rest=pr.drop(pres)
    hi=rest[rest.R>=4]; lo=rest[rest.R<4]
    for i,r in hi.iterrows(): nm[i]='Esfriando (alto valor)' if r.F>=6 else 'Quase perdidos'
    for i,r in lo[lo.F<6].iterrows(): nm[i]='Novatos em experimentação'
    top=lo[lo.F>=6].sort_values('cupom')
    if len(top)==2: nm[top.index[0]]='Fiéis a preço cheio'; nm[top.index[1]]='Cinéfilos de cupom'
    assert len(set(nm.values()))==len(nm)==K, nm
    return d[g].map(nm)
p['grupo']=nomear(p,'g')
tab=p.groupby('grupo').agg(clientes=('R','size'),receita_total=('M','sum'),receita_media=('M','mean'),cancel_90d=('churn90','mean'),R_mediana=('R','median'),F_mediana=('F','median'),
    cupons_medios=('cupom','mean'),pct_presente=('presente','mean'),nps_medio=('nps','mean'),pct_midia_paga=('grupo_canal',lambda s:(s=='Mídia paga').mean()),pct_cartao=('origem',lambda s:(s=='Internet (cartão)').mean()))
tab['pct_clientes']=tab.clientes/tab.clientes.sum(); tab['pct_receita']=tab.receita_total/tab.receita_total.sum()
canal=p.groupby('grupo').canal_aquisicao.agg(lambda s:s.value_counts().index[0]); canal_pct=p.groupby('grupo').canal_aquisicao.agg(lambda s:s.value_counts(normalize=True).iloc[0])
tab['canal_mais_comum']=canal; tab['pct_do_canal']=canal_pct
ct_=pd.crosstab(p.grupo,p.canal_aquisicao,normalize='index'); lift=ct_/p.canal_aquisicao.value_counts(normalize=True)
tab['canal_maior_lift']=lift.idxmax(axis=1); tab['lift']=lift.max(axis=1)
tab=tab.sort_values('pct_receita',ascending=False)
# ---------- 3) ESTABILIDADE DO MODELO FINAL
ev=rng.choice(len(Z),100000,replace=False); ref=p.iloc[ev].grupo.values; agree=[]; aris=[]
pe=p.iloc[ev].copy()
for s in range(5):
    ids=np.random.default_rng(200+s).choice(len(Z),100000,replace=False)
    kms=KMeans(K,n_init=5,random_state=s).fit(Z[ids]); pe['g2']=kms.predict(Z[ev]); n2=nomear(pe,'g2').values
    agree.append((n2==ref).mean()); aris.append(adjusted_rand_score(p.iloc[ev].g.values,pe.g2.values))
# sem NPS (na vida real so ha respondentes) e sem presente
def refit(cols_idx):
    km2=KMeans(K,n_init=5,random_state=1).fit(Zs[:,cols_idx]); return km2.predict(Z[ev][:,cols_idx])
sem_nps=adjusted_rand_score(p.iloc[ev].g.values,refit([0,1,2,3,4])); sem_pres=adjusted_rand_score(p.iloc[ev].g.values,refit([0,1,2,3,5]))
stab_final=dict(concordancia_nome_media=np.mean(agree),concordancia_nome_min=np.min(agree),ARI_medio=np.mean(aris),ARI_sem_nps=sem_nps,ARI_sem_presente=sem_pres)
# ---------- 4) CANCELAMENTO POR CANAL (base inteira, 24 meses)
ch=df.groupby(['grupo_canal','origem','canal_aquisicao'],observed=True).agg(assinaturas=('canc','size'),cancelados=('canc','sum'),ciclos_cobrados=('meses_cobrados','sum')).reset_index()
ch['pct_cancelados_acum']=ch.cancelados/ch.assinaturas; ch['churn_mensal']=ch.cancelados/ch.ciclos_cobrados
ch=ch.sort_values('churn_mensal',ascending=False)
# FIX 3: removido loop vazio "for col in ['grupo_canal','origem']: pass"
ch_grp=df.groupby(['origem','grupo_canal'],observed=True).agg(assinaturas=('canc','size'),cancelados=('canc','sum'),ciclos=('meses_cobrados','sum')).reset_index(); ch_grp['pct_cancelados_acum']=ch_grp.cancelados/ch_grp.assinaturas; ch_grp['churn_mensal']=ch_grp.cancelados/ch_grp.ciclos
# ---------- 5) CONCENTRACAO DE RECEITA
def top_share(v,q=.2):
    v=np.sort(np.asarray(v))[::-1]; n=int(np.ceil(len(v)*q)); return v[:n].sum()/v.sum()
conc=pd.DataFrame({'base':['Todas as 1,92 mi assinaturas (receita acumulada em 24m)','Base ativa em 26/07/2026 (valor pago até a data)'],
  'top_1pct':[top_share(df.receita_total_paga,.01),top_share(p.M,.01)],'top_10pct':[top_share(df.receita_total_paga,.10),top_share(p.M,.10)],'top_20pct':[top_share(df.receita_total_paga),top_share(p.M)],'top_50pct':[top_share(df.receita_total_paga,.5),top_share(p.M,.5)]})
# ---------- 6) APLICAR NA BASE ATUAL (usar amanha): mesmos escaladores e centroides
a=df[df.status=='Ativo'].copy()
# FIX 4: recalcular F, M e cupom com a mesma lógica de treino (seção RFM)
#         versão anterior usava n_pedidos/receita_total_paga → definições diferentes → atribuições de segmento erradas
ten_a=((SNAP_D-a.start_d)//30+1).astype(float); frac_a=np.minimum(1,ten_a/a.meses_cobrados.replace(0,1))
a['R']=a.R_dias_sem_acesso.astype(float)
a['F']=ten_a+(a.tvod_alugueis+a.tvod_compras)*frac_a
a['M']=ten_a*a.preco_lista_mensal-np.minimum(ten_a,2)*a.preco_lista_mensal*0.4*a.cupom_entrada+a.receita_tvod*frac_a
a['cupom']=a.cupom_entrada+a.cupons_tvod_usados*frac_a
Za=sc.transform(prep(a)).astype('float32'); a['g']=km.predict(Za)
# FIX 5: drop_duplicates garante exatamente uma entrada por cluster (dict(zip) iterava todas as linhas)
mapa=p.drop_duplicates('g').set_index('g')['grupo'].to_dict(); a['grupo']=a.g.map(mapa)
atual=a.groupby('grupo').agg(clientes=('R','size'),receita_acum=('M','sum')); atual['pct_clientes']=atual.clientes/atual.clientes.sum(); atual['pct_receita']=atual.receita_acum/atual.receita_acum.sum()
a[['id_cliente','origem','plano','grupo_canal','canal_aquisicao','R','F','M','cupom','presente','nps','grupo']].rename(columns={'R':'recencia_dias','F':'pedidos','M':'valor_pago','cupom':'cupons'}).to_csv(OUT_CSV,index=False,compression={'method':'gzip','compresslevel':3})
pickle.dump(dict(expl=expl,sel=sel,stab=stab,nest=nest,tab=tab,stab_final=stab_final,ch=ch,ch_grp=ch_grp,conc=conc,atual=atual,centros=pd.DataFrame(sc.inverse_transform(km.cluster_centers_),columns=FEATS).assign(grupo=lambda d:d.index.map(mapa))),open(os.path.join(PKL_DIR,'stage2.pkl'),'wb'))
print(tab.round(3).T.to_string()); print(stab_final); print(ch.round(4).to_string()); print(ch_grp.round(4)); print(conc.round(3).to_string()); print(atual.round(3))

# ---------- 7) FILTRO POR PLANO e CUPONS POR PLANO
tab_plano=p.groupby(['plano','grupo'],observed=True).agg(clientes=('R','size'),receita_total=('M','sum'),churners_90d=('churn90','sum'),cupons=('cupom','sum')).reset_index()
atual_plano=a.groupby(['plano','grupo'],observed=True).agg(clientes=('R','size'),valor_pago=('M','sum')).reset_index()
# FIX 6: removido pct_usa_tvod=lambda s:np.nan do agg — coluna era criada como NaN e sobrescrita na linha seguinte
cup=df.groupby('plano',observed=True).agg(assinaturas=('canc','size'),pct_cupom_entrada=('cupom_entrada','mean'),alugueis_por_assinatura=('tvod_alugueis','mean'),cupons_tvod_por_assinatura=('cupons_tvod_usados','mean'),
    receita_tvod=('receita_tvod','sum'),receita_total=('receita_total_paga','sum')).reset_index()
# include_groups=False suprime DeprecationWarning do pandas 2.2+ (groupby key não passa ao lambda)
cup['pct_usa_tvod']=df.groupby('plano',observed=True).apply(lambda g:((g.tvod_alugueis+g.tvod_compras)>0).mean(),include_groups=False).values
cup['pct_alugueis_com_cupom']=df.groupby('plano',observed=True).apply(lambda g:g.cupons_tvod_usados.sum()/g.tvod_alugueis.sum(),include_groups=False).values
cup['pct_receita_tvod']=cup.receita_tvod/cup.receita_total
PRICE={'Premiere Futebol':59.90,'Telecine':37.90,'Plano Família Claro tv+':99.90}; CMg={'Premiere Futebol':.45,'Telecine':.58,'Plano Família Claro tv+':.38}
# FIX 7: MARGEM {preco→margem} elimina a comparação float frágil abs(v-59.9)<.01 em eff()
MARGEM={v:CMg[k] for k,v in PRICE.items()}
cup['plano']=cup.plano.astype(str); cup['preco']=cup.plano.map(PRICE).astype(float); cup['margem_pct']=cup.plano.map(CMg).astype(float)
cup['custo_desconto_2m']=cup.preco*0.40*2; cup['contrib_mensal_cheia']=cup.preco*cup.margem_pct
cup['margem_nos_meses_promo_pct_do_preco']=0.60-(1-cup.margem_pct)
cup['meses_para_pagar_desconto']=cup.custo_desconto_2m/cup.contrib_mensal_cheia
# efeito observado do cupom de entrada (dentro de estratos origem x aquisicao)
def eff(g):
    r=[]
    for (o,gc),h in g.groupby(['origem','grupo_canal'],observed=True):
        for k in (3,12):
            ob=h[h.start_d+30*k<=SNAP_D]; pr=ob[ob.cupom_entrada==1]; np_=ob[ob.cupom_entrada==0]
            r.append(dict(origem=o,grupo=gc,k=k,w=len(pr),ret_com=(pr.meses_cobrados>k).mean(),ret_sem=(np_.meses_cobrados>k).mean()))
    r=pd.DataFrame(r); out={}
    for k in (3,12):
        x=r[r.k==k]; out[f'ret_{k}m_com']=np.average(x.ret_com,weights=x.w); out[f'ret_{k}m_sem']=np.average(x.ret_sem,weights=x.w)
    ob=g[g.start_d+360<=SNAP_D]; n12=np.minimum(ob.meses_cobrados,12).astype(float); price=ob.preco_lista_mensal
    # FIX 7 (cont.): price.map(MARGEM) substitui abs(v-59.9)<.01 — robusto a variações de float
    ob=ob.assign(m12=(n12*price-np.minimum(n12,2)*price*0.4*ob.cupom_entrada)*price.map(MARGEM))
    rr=[]
    for (o,gc),h in ob.groupby(['origem','grupo_canal'],observed=True): rr.append((len(h[h.cupom_entrada==1]),h[h.cupom_entrada==1].m12.mean(),h[h.cupom_entrada==0].m12.mean()))
    rr=np.array(rr); out['margem12_com']=np.average(rr[:,1],weights=rr[:,0]); out['margem12_sem']=np.average(rr[:,2],weights=rr[:,0])
    return pd.Series(out)
cup_eff=df.groupby('plano',observed=True).apply(eff,include_groups=False).reset_index()
pickle.dump(dict(tab_plano=tab_plano,atual_plano=atual_plano,cup=cup,cup_eff=cup_eff),open(os.path.join(PKL_DIR,'stage3.pkl'),'wb'))
print(tab_plano.assign(p=lambda d:d.clientes/d.groupby('plano').clientes.transform('sum')).pivot(index='grupo',columns='plano',values='p').round(3).to_string()); print(cup.round(3).T.to_string()); print(cup_eff.round(3).T.to_string())
