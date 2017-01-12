A machine-learning based truth recogniser. This code is similar to
the expression evaluator in test2.lhs so see that for details.

In this case the task is to recognise true propositions of propositional
calculus using the connectives &, |, -> as well as ~, two constants
0 and 1 and two variables a and b.

Usually if there are free variables in a proposition it's considered
true if it's true whatever truth values are assigned to a and b.
I'm using a slightly modified form: a proposition is considered
true if whatever truth value is assigned to a we can find a truth
value to assign to b to make the propostion true. This is purely
because it results in a more balanced training set.

Note that truth in this sense is not trivially analytic in the sense
that you can't tell if X->Y, say, is true, simply by knowing whether
X and Y are true.  Consider that neither "a" and "0" are true in
the current framework yet "(0->a)" is not true but both "(0->0)"
and "(a->a)" are.

The code is being trained purely on strings. It knows nothing about
subexpressions, ir that "(" signals a subexpression, or that "("
isn't a connective, or that "a" and "b: are variables. or that "->"
is a two character connective.

Note that the code gets to see a large proportion of the possible
expressions.  But it can't be "overlearning", it only has about
a few thousand weights.

> {-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
> {-# LANGUAGE NoMonomorphismRestriction, DeriveGeneric #-}
> 
> import Prelude hiding (init, negate, error)
> import System.Random
> import Control.Monad.State
> import Control.DeepSeq
> import Data.Hash
> import qualified GHC.Generics as G
> import qualified Data.Foldable as F
> import qualified Data.Traversable as T
> import qualified Data.Vector as V
> import Numeric.AD.Mode.Reverse as R
> 
> dot :: Num a => V.Vector a -> V.Vector a -> a
> dot u v = V.sum $ V.zipWith (*) u v
> multMV :: Num a => V.Vector (V.Vector a) -> V.Vector a -> V.Vector a
> multMV a x = V.fromList [dot r x | r <- V.toList a]
> 
> type Matrix a = V.Vector (V.Vector a)
> data LogicNet a = A { init :: V.Vector a
>                     , vara :: Matrix a
>                     , varb :: Matrix a
>                     , conjunction :: Matrix a
>                     , disjunction :: Matrix a
>                     , minus :: Matrix a
>                     , negate :: Matrix a
>                     , gt :: Matrix a
>                     , lparen :: Matrix a
>                     , rparen :: Matrix a
>                     , false :: Matrix a
>                     , true :: Matrix a
>                     , fini :: V.Vector a
>                     , fini_b :: a
>                     } deriving (Show, Functor, F.Foldable,
>                                 T.Traversable, G.Generic)
> instance NFData a => NFData (LogicNet a)
> 
> randomVector :: (RandomGen g, MonadState g m) =>
>                 Double -> Int -> m (V.Vector Double)
> randomVector r n = fmap V.fromList $ replicateM n $
>                       state (randomR (-r, r))
> randomMatrix :: (RandomGen g, MonadState g m) =>
>                 Double -> Int -> Int -> m (Matrix Double)
> randomMatrix r m n = fmap V.fromList $ replicateM m (randomVector r n)
> randomLogicNet :: (Applicative m, RandomGen g, MonadState g m) =>
>                   Int -> m (LogicNet Double)
> randomLogicNet d = A <$> randomVector 0.5 d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (2.5/sqrt (fromIntegral d)) d d
>                      <*> randomVector 0.5 d
>                      <*> pure 0.0
> update :: Num a => LogicNet a -> a -> LogicNet a -> LogicNet a
> update (A a b c d e f g h i j k l y z) r (A a' b' c' d' e' f' g' h' i' j' k' l' y' z') =
>               A (V.zipWith (+) a (V.map (r *) a'))
>                 (V.zipWith (V.zipWith (+)) b (V.map (V.map (r *)) b'))
>                 (V.zipWith (V.zipWith (+)) c (V.map (V.map (r *)) c'))
>                 (V.zipWith (V.zipWith (+)) d (V.map (V.map (r *)) d'))
>                 (V.zipWith (V.zipWith (+)) e (V.map (V.map (r *)) e'))
>                 (V.zipWith (V.zipWith (+)) f (V.map (V.map (r *)) f'))
>                 (V.zipWith (V.zipWith (+)) g (V.map (V.map (r *)) g'))
>                 (V.zipWith (V.zipWith (+)) h (V.map (V.map (r *)) h'))
>                 (V.zipWith (V.zipWith (+)) i (V.map (V.map (r *)) i'))
>                 (V.zipWith (V.zipWith (+)) j (V.map (V.map (r *)) j'))
>                 (V.zipWith (V.zipWith (+)) k (V.map (V.map (r *)) k'))
>                 (V.zipWith (V.zipWith (+)) l (V.map (V.map (r *)) l'))
>                 (V.zipWith (+) y (V.map (r *) y'))
>                 (z+r*z')
> data Expr = VarA | VarB | Bottom | Top | And Expr Expr | Or Expr Expr | Implies Expr Expr | Not Expr
> instance Show Expr where
>     show VarA = "a"
>     show VarB = "b"
>     show Bottom = "0"
>     show Top = "1"
>     show (And a b) = "(" ++ show a ++ "&" ++ show b ++ ")"
>     show (Or a b) = "(" ++ show a ++ "|" ++ show b ++ ")"
>     show (Implies a b) = "(" ++ show a ++ "->" ++ show b ++ ")"
>     show (Not a) = "(~" ++ show a ++ ")"
> 
> relu :: (Num a, Ord a) => a -> a
> relu x | x < 0 = 0
> relu x = x
> 
> activation :: (Floating a, Ord a) => a -> a
> activation x = relu x
> 
> run :: (Floating b, Ord b) => LogicNet b -> String -> V.Vector b -> V.Vector b
> run _ "" v = v
> run net ('(' : as) v = run net as (V.map activation $ multMV (lparen net) v)
> run net (')' : as) v = run net as (V.map activation $ multMV (rparen net) v)
> run net ('-' : as) v = run net as (V.map activation $ multMV (minus net) v)
> run net ('>' : as) v = run net as (V.map activation $ multMV (gt net) v)
> run net ('0' : as) v = run net as (V.map activation $ multMV (false net) v)
> run net ('1' : as) v = run net as (V.map activation $ multMV (true net) v)
> run net ('&' : as) v = run net as (V.map activation $ multMV (conjunction net) v)
> run net ('~' : as) v = run net as (V.map activation $ multMV (negate net) v)
> run net ('|' : as) v = run net as (V.map activation $ multMV (disjunction net) v)
> run net ('a' : as) v = run net as (V.map activation $ multMV (vara net) v)
> run net ('b' : as) v = run net as (V.map activation $ multMV (varb net) v)
> 
> error' :: Bool -> Double -> Double
> error' a b = if a then 1.0-b else 0.0-b
> 
> error :: Bool -> Double -> Double
> error a b = if a then 1.0/b else -1.0/(1.0-b)
> 
> sigmoid :: Floating a => a -> a
> sigmoid x = 1.0/(1.0+exp (-x))
> 
> full :: (Floating a, Ord a) => LogicNet a -> String -> a
> full net expr = sigmoid $ fini_b net+dot (fini net) (run net expr (init net))
> 
> learn :: (Expr -> Bool) -> Double -> Int -> Int -> (Expr -> Bool) ->
>          Int -> LogicNet Double -> StateT StdGen IO (LogicNet Double)
> learn _ _ _ _ _ 0 net = return net
> learn cond learningRate minSize maxSize eval n net = do
>     size <- state $ randomR (minSize, maxSize :: Int)
>     expr <- randomE size
>     if cond expr
>       then do
>        when (n `mod` 1000==0) $ liftIO $ print n
>        let value = eval expr
>        let flat = show expr
>        let value' = full net flat 
>        when (n `mod` 1000==0) $
>           liftIO $ putStrLn $ show expr ++ " " ++ show value ++ " " ++ show value'
>        let g = grad (flip full flat) net
>        let net' = update net (learningRate*(error value value')) g
>        net' `deepseq` learn cond learningRate minSize maxSize eval (n-1) net'
>       else learn cond learningRate minSize maxSize eval n net
> 
> test :: (Expr -> Bool) -> Int -> Int -> (Expr -> Bool) ->
>          Int -> LogicNet Double -> StateT StdGen IO ()
> test _ _ _ _ 0 _ = return ()
> 
> test cond minSize maxSize eval n net = do
>     size <- state $ randomR (minSize, maxSize :: Int)
>     expr <- randomE size
>     if cond expr
>       then do
>        let value = eval expr
>        let flat = show expr
>        let value' = full net flat
>        liftIO $ putStrLn $ show expr ++ " " ++ show value ++ " " ++ show value'
>        test cond minSize maxSize eval (n-1) net
>       else test cond minSize maxSize eval n net
> 
> randomE :: (RandomGen g, MonadState g m) => Int -> m Expr
> randomE 1 = do
>     variable <- state $ randomR (0, 11 :: Int)
>     case variable of
>         0 -> return Bottom
>         1 -> return Top
>         2 -> return VarA
>         3 -> return VarA
>         4 -> return VarA
>         5 -> return VarA
>         6 -> return VarA
>         7 -> return VarB
>         8 -> return VarB
>         9 -> return VarB
>         10 -> return VarB
>         11 -> return VarB
> 
> randomE size = do
>     op <- state $ randomR (0, 3 :: Int)
>     case op of
>         0 -> do
>             childSize <- state $ randomR (1, size-1)
>             a <- randomE childSize
>             b <- randomE (size-childSize)
>             return $ Implies a b
>         1 -> do
>             childSize <- state $ randomR (1, size-1)
>             a <- randomE childSize
>             b <- randomE (size-childSize)
>             return $ And a b
>         2 -> do
>             childSize <- state $ randomR (1, size-1)
>             a <- randomE childSize
>             b <- randomE (size-childSize)
>             return $ Or a b
>         3 -> do
>             a <- randomE (size-1)
>             return $ Not a
> 
> num :: Int -> Int

There are two variables and two constants that can appear as leaves:

> num 1 = 4

And three connectives as well as not:

> num size = sum [3*num i*num (size-i) + num (size-1) | i <- [1..(size-1)]]
> 
> num' :: Int -> Int -> Int
> num' minSize maxSize = sum [num i | i <- [minSize..maxSize]]
> 
> eval1' :: (Bool, Bool) -> Expr -> Bool
> eval1' _ Bottom = False
> eval1' _ Top = True
> eval1' (a, _) VarA = a
> eval1' (_, b) VarB = b
> eval1' e (And a b) = eval1' e a && eval1' e b
> eval1' e (Or a b) = eval1' e a || eval1' e b
> eval1' e (Implies a b) = not (eval1' e a) || eval1' e b
> eval1' e (Not a) = not (eval1' e a)
> 
> eval1 :: Expr -> Bool
> eval1 e = (eval1' (False, False) e || eval1' (False, True) e)
>           &&
>           (eval1' (True, False) e || eval1' (True, True) e)
> 
> main :: IO ()
> main = do
>     let dimension = 18 :: Int
>     let minSize = 2 :: Int
>     let maxSize = 6 :: Int
>     let learningRate = 0.0001
>     putStrLn $ "There are " ++ show (num' minSize maxSize) ++
>                " expressions of size " ++ show minSize ++ " to " ++ show maxSize
>     let gen = mkStdGen 99
>     flip evalStateT gen $ do
>         net <- randomLogicNet dimension
>         net' <- learn (even . asWord64 . hash . show) learningRate minSize maxSize eval1 5000000 net
>         liftIO $ print "Testing training set"
>         test (even . asWord64 . hash . show) minSize maxSize eval1 100 net'
>         liftIO $ print "Testing test set"
>         test (odd . asWord64 . hash . show) minSize maxSize eval1 100 net'

Results from a sample run
For each row is the expression, whether or not it's true
(by my definition above) and the score assigned to it.
I sorted on the score to make it easy to see the errors
it made.

(b&a)                     False  2.2325651166354522e-14
(~a)                      False  1.597580854771411e-10
(~a)                      False  1.597580854771411e-10
(~a)                      False  1.597580854771411e-10
((b&b)&0)                 False  1.0127962038298551e-8
(0&b)                     False  1.7011236256143674e-5
((a&b)&b)                 False  2.391044641538165e-5
((b->0)&a)                False  8.251529764308685e-5
(b->a)                    True   3.4910525769579715e-4
(b->a)                    True   3.4910525769579715e-4
(a&a)                     False  4.356196886196246e-4
(a&a)                     False  4.356196886196246e-4
(a&a)                     False  4.356196886196246e-4
(a&a)                     False  4.356196886196246e-4
(a&b)                     False  4.924899116108857e-4
((b|(a&a))&a)             False  5.445763478187471e-4
((a&a)&b)                 False  5.874062449120694e-4
(~(~a))                   False  1.0383042729318788e-3
(~(~a))                   False  1.0383042729318788e-3
(~(~a))                   False  1.0383042729318788e-3
(~(~a))                   False  1.0383042729318788e-3
(((a|b)|b)&(~a))          False  1.4400978972243241e-3
((~((a->(b->a))->a))&a)   False  2.7646521999396254e-3
(((b->(a->(a|b)))&a)&b)   False  3.7775663093982566e-3
((~(~a))&b)               False  7.1746247051947655e-3
((b&1)&(a|a))             False  7.4474438547081166e-3
((~((1|b)->a))&0)         False  9.772055675827125e-3
(a|(b&a))                 False  1.5250493131140219e-2
((~((0&(a->a))|a))&a)     False  2.3080006781095418e-2
(((a->a)->a)&a)           False  2.661170299007897e-2
(a|0)                     False  4.406935542183463e-2
((a&a)&(1&a))             False  4.846144282536141e-2
(0&(((a->b)|b)|a))        False  5.629382626405571e-2
(0&((~((~1)->b))&a))      False  5.629382626405571e-2
(a&((~0)|(~a)))           False  5.629382626405571e-2
(a&(a&a))                 False  5.629382626405571e-2
(a&(b|b))                 False  5.629382626405571e-2
(a&(~(1|(~a))))           False  5.629382626405571e-2
(a&(~(a->(b&(0->b)))))    False  5.629382626405571e-2
(a&(~a))                  False  5.629382626405571e-2
((a&b)&((a|a)&b))         False  5.700152601814039e-2
(~(a|((~a)&a)))           False  6.850777069575119e-2
((a&a)|(~a))              True   6.923837681465458e-2
((~(a|a))&(~(a->a)))      False  7.800651390146746e-2
(~(a->(a&b)))             False  8.083822540546129e-2
(~(a->(b|a)))             False  8.543994581874914e-2
(~(a->b))                 False  8.600938331132701e-2
(~(~(~(a->(a&a)))))       False  9.083358902696818e-2
(~(0->0))                 False  9.246942861295351e-2
((a->b)->(~a))            True   9.515714226893328e-2
(a->(b&0))                False  0.10927559941405539
(~(((a&b)&1)->(a->a)))    False  0.11025383835246874
(((a->b)|0)->a)           False  0.11461362715992164
(~((~(~(b->1)))&a))       False  0.13129242371142097
(((b|b)&(a&(b&1)))->a)    True   0.1337498991985547
((~(a->b))->0)            True   0.15270695613360147
(~(a&a))                  False  0.15493023341518256
(~(~(a&(~(a|a)))))        False  0.15766818885219638
((a|(b&a))&(~(b|a)))      False  0.16509102560905936
(~(b|(((0|a)|a)->a)))     False  0.16899330473746668
(~((b->b)->(b->a)))       False  0.19469991921715718
(~(1|(1|(~a))))           False  0.19715562485468074
(~((~1)|((a->b)&a)))      True   0.2046093850970516
(~((a&a)|b))              False  0.21167040423393374
(~(b->((~a)|(b&b))))      False  0.2205555751640541
(((0->0)|b)&((~a)&a))     False  0.25247285796714986
(b&((~((a&a)&a))&b))      False  0.25953067890604414
(a|(a&(b&(a->a))))        False  0.264446400771597
((b->b)->a)               False  0.2779262448487086
(1&(~((a&b)&a)))          True   0.2796487756956938
(~(((~b)->a)&(1|1)))      False  0.2928029208024307
(((a|a)->1)&(0|(~a)))     False  0.3004140111582713
(~(~(~(b->b))))           False  0.302203504797726
(((~0)->b)->(~(b|a)))     True   0.30403236603543693
(1&(~(a->(a&a))))         False  0.31219091751253714
(~(0|(b->a)))             False  0.3418423012219008
(~(~(~(b->(b&a)))))       False  0.3463049583769368
((a&(1|a))|((1&a)&a))     False  0.36822204185075635
(~(((a|b)->1)&b))         True   0.40638792730742196
(a|(a|a))                 False  0.4119456241114502
((~((b->0)&b))|a)         True   0.4902915556792344
(1&(((~a)&b)|a))          True   0.5100951925216171
((b|a)->(0&(~(1|a))))     False  0.5348143722570894
(a->(b|a))                True   0.539694200986275
(((~1)|(a&b))->a)         True   0.542449682178916
(a|a)                     False  0.5594557925052926
(a|a)                     False  0.5594557925052926
((~(b|(b|b)))&b)          False  0.5795523699631268
(a|(~((a->a)|1)))         False  0.5838334301546739
((a&(~b))->(a&(~b)))      True   0.5898295302025321
(~((b&a)|(0|b)))          True   0.5967108956466383
((~b)&(b->a))             True   0.6470552794004988
(~((~b)->b))              True   0.694602414257272
(~(~((a|a)->a)))          True   0.7144809059509387
((~(b->(1|b)))->(a|a))    True   0.7399509879632618
(~(((b|b)|(a->0))->b))    False  0.7559195353788916
((b->(1|0))&(b&(a|b)))    True   0.7748089871718625
((b->((~a)&a))|(0|a))     True   0.8203570831493808
(a|(~((a->b)|(b|a))))     False  0.8234188005135226
((b|(~a))|(~(~0)))        True   0.826579042892251
(~(1->b))                 True   0.8288448642961156
(~(a&(b&a)))              True   0.836525130397505
((((b&b)&1)&b)|a)         True   0.8414696443747866
(~(~(a->a)))              True   0.849737554950559
(b|(a|a))                 True   0.855509957583863
(((0|b)->a)|((~a)&b))     True   0.8691189369346556
(a->(b&a))                True   0.9130702999475849
((1&(b->1))|a)            True   0.9165380600472524
(a|(b&((~b)|(~1))))       False  0.9235814552799944
(a->(a|0))                True   0.9448692786377767
(~((~b)&(b&b)))           True   0.9476199051560137
(b&((b&(b&b))|b))         True   0.9512811878147627
(a|((a&a)->b))            True   0.9649727764394516
((b|(b|b))&(a|(~a)))      True   0.9659027867585555
(1->((a->b)|a))           True   0.9769913206548448
(~(b&(0|(~1))))           True   0.9785659103813826
((b->a)->(~((b&b)&b)))    True   0.9805235295081559
(~(~0))                   False  0.9824709303549346
(((a->a)->(~b))|a)        True   0.9826941638404934
((a|b)&b)                 True   0.9839182715761531
((b|a)&(a->a))            True   0.9879398393777946
((b|a)|a)                 True   0.9880806203599128
((~b)|(a&b))              True   0.9888949570812336
(((~b)->b)&((a&b)|b))     True   0.9908159656085659
(b|(~(b|a)))              True   0.9922958600774452
(~(~(~b)))                True   0.9927055965033377
(~(b&(0|(b->a))))         True   0.9929044781822965
((~a)|a)                  True   0.9936493863820505
(~b)                      True   0.9943211696530861
(~b)                      True   0.9943211696530861
(~b)                      True   0.9943211696530861
(((b->a)|1)|(a&1))        True   0.9950935933393023
(b|(a&a))                 True   0.9951718372552166
(b&(a|b))                 True   0.9953708265946928
((a|(~(1&0)))|(a|b))      True   0.9989003348450347
((~(a->(b->a)))->1)       True   0.9989247882504759
(a|(b|(b&a)))             True   0.9990082610177504
((~a)|(1->b))             True   0.9991934563721472
(((~(a&(b&a)))|b)->b)     True   0.9992655445120859
(~(b&(~(a&b))))           True   0.9992926099122799
((a->b)|((a|b)|a))        True   0.9993398220843346
(a|(0->a))                True   0.9993481240499241
((b|1)->(a->b))           True   0.9993485405293421
((b|b)|(~((1&1)|a)))      True   0.9993828646885726
(~0)                      True   0.9996440723375002
(a|(a->((~b)|1)))         True   0.9998136891158734
(1->1)                    True   0.9998170997155869
(1&b)                     True   0.9998333300567832
(b&b)                     True   0.9998841965405258
(b&b)                     True   0.9998841965405258
(b&b)                     True   0.9998841965405258
(b&b)                     True   0.9998841965405258
(b->(b->(a|a)))           True   0.9999009310656738
(~(b&((b&a)&b)))          True   0.9999262447948539
((b->(~a))|(a|b))         True   0.999931845763647
(b->0)                    True   0.9999376523818553
(b->0)                    True   0.9999376523818553
(((a|0)&(b->a))->b)       True   0.9999378558884788
((a&(b->a))->b)           True   0.9999424487813001
(~(~b))                   True   0.9999431975099035
((b|(~b))|((a&b)&1))      True   0.9999498390442585
(a|b)                     True   0.9999523553708966
((a&b)->(b|0))            True   0.9999741938886155
((a->b)|1)                True   0.9999819564408526
(((a|b)->0)|(b|1))        True   0.9999894092708781
(((~(b|0))->b)|a)         True   0.9999911921204249
((~(((~b)->b)->a))|b)     True   0.9999927493381534
(~(~(b|(b&b))))           True   0.9999935375896325
(b|(~0))                  True   0.9999946571779011
((~(~((b&b)->a)))|b)      True   0.9999960310389007
((a->(~b))|(b|a))         True   0.9999961553325893
((((b&b)|a)|b)->(b|b))    True   0.9999963908925819
((a|1)|b)                 True   0.9999988889948217
((~b)->((~b)|a))          True   0.9999990750903048
(a->a)                    True   0.9999994692881834
((~((~0)|(b|a)))->b)      True   0.9999995176441929
(((0|b)&(b|(~a)))->b)     True   0.9999999295569225
((a->0)->b)               True   0.999999979591635
((a->b)|b)                True   0.9999999849993051
(0->(a|1))                True   0.9999999907086553
(((1&b)->(b->a))|(a->b))  True   0.9999999955452583
(b|1)                     True   0.9999999985697028
(b|1)                     True   0.9999999985697028
(b|1)                     True   0.9999999985697028
(b|b)                     True   0.9999999996179301
(b|(b|a))                 True   0.9999999998141929
(b->(1|b))                True   0.9999999999601197
(a->b)                    True   0.9999999999770526
((b->a)|b)                True   0.9999999999795761
(1|b)                     True   0.9999999999942304
((a&b)->b)                True   0.9999999999948106
(b|(b&b))                 True   0.9999999999965397
(b|((~a)|(a->b)))         True   0.9999999999996545
(b|(~b))                  True   0.9999999999999802
((~b)->b)                 True   0.9999999999999982
(((b|1)|(~a))->b)         True   1.0
((b&1)|b)                 True   1.0
(b->1)                    True   1.0
(b->1)                    True   1.0
(b->b)                    True   1.0
