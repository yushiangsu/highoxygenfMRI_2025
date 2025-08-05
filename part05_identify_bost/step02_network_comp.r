library(BayesFactor)

dat  = read.csv("./rois_betas.tsv", sep = '\t')

net_list = c("Default", "SalVentAttn", "SomMot", "Cont", "DorsAttn", "Limbic", "Vis")
for (i in seq(1, 6)) {
  for (j in seq(i+1, 7)) {
    print(paste0(net_list[i], "_", net_list[j]))
    t.test(dat$p2p1_psc[dat$network == net_list[i]], 
           dat$p2p1_psc[dat$network == net_list[j]])
  }
}
t.test(dat$p2p1_psc[dat$network == "Default"], 
       dat$p2p1_psc[dat$network == "Cont"], var.equal = T)
ttestBF(dat$p2p1_psc[dat$network == "Default"], 
        dat$p2p1_psc[dat$network == "Cont"])

t.test(dat$p4p1_psc[dat$network == "Default"], 
       dat$p4p1_psc[dat$network == "Cont"], var.equal = T)
ttestBF(dat$p4p1_psc[dat$network == "Default"], 
        dat$p4p1_psc[dat$network == "Cont"])
