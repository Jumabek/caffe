 ./build/tools/caffe train --solver=models/bvlc_reference_caffenet_siamese_5conv/solver.prototxt 2> models/bvlc_reference_caffenet_siamese_5conv/results.log

 ./build/tools/caffe train -solver models/bvlc_reference_caffenet_siamese_5conv/solver.prototxt  -snapshot models/bvlc_reference_caffenet_siamese_5conv/caffenet_train_iter_450000.solverstate 2> models/bvlc_reference_caffenet_siamese_5conv/results2.log

