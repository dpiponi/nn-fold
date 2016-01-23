> {-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
> {-# LANGUAGE NoMonomorphismRestriction, DeriveGeneric #-}

> import Prelude hiding (init)
> import System.Random
> import Control.Monad.State
> import Control.DeepSeq
> import Control.Applicative
> import Data.Hash

> import qualified GHC.Generics as G
> import qualified Data.Foldable as F
> import qualified Data.Traversable as T
> import qualified Data.Vector as V

> import Numeric.AD.Mode.Reverse as R

> dot :: Num a => V.Vector a -> V.Vector a -> a
> dot u v = V.sum $ V.zipWith (*) u v

> multMV :: Num a => V.Vector (V.Vector a) -> V.Vector a -> V.Vector a
> multMV a x = V.fromList [dot r x | r <- V.toList a]

> type Matrix a = V.Vector (V.Vector a)

> data ArithNet a = A { init :: V.Vector a
>                     , zero :: Matrix a
>                     , one :: Matrix a
>                     , plus :: Matrix a
>                     , times :: Matrix a
>                     , lparen :: Matrix a
>                     , rparen :: Matrix a
>                     , fini :: V.Vector a
>                     } deriving (Show, Functor, F.Foldable,
>                                 T.Traversable, G.Generic)

> instance NFData a => NFData (ArithNet a)

> randomVector :: (Functor m, RandomGen g, MonadState g m) =>
>                 Double -> Int -> m (V.Vector Double)
> randomVector r n = fmap V.fromList $ replicateM n $
>                       state (randomR (-r, r))

> randomMatrix :: (Functor m, RandomGen g, MonadState g m) =>
>                 Double -> Int -> Int -> m (Matrix Double)
> randomMatrix r m n = fmap V.fromList $ replicateM m (randomVector r n)

> randomArithNet :: (Applicative m, Functor m, RandomGen g, MonadState g m) =>
>                   Int -> m (ArithNet Double)

> randomArithNet d = A <$> randomVector 0.5 d
>                      <*> randomMatrix (3/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (3/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (3/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (3/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (3/sqrt (fromIntegral d)) d d
>                      <*> randomMatrix (3/sqrt (fromIntegral d)) d d
>                      <*> randomVector 0.5 d

> update :: Num a => ArithNet a -> a -> ArithNet a -> ArithNet a
> update (A a b c d e f g h) r (A a' b' c' d' e' f' g' h') =
>               A (V.zipWith (+) a (V.map (r *) a'))
>                 (V.zipWith (V.zipWith (+)) b (V.map (V.map (r *)) b'))
>                 (V.zipWith (V.zipWith (+)) c (V.map (V.map (r *)) c'))
>                 (V.zipWith (V.zipWith (+)) d (V.map (V.map (r *)) d'))
>                 (V.zipWith (V.zipWith (+)) e (V.map (V.map (r *)) e'))
>                 (V.zipWith (V.zipWith (+)) f (V.map (V.map (r *)) f'))
>                 (V.zipWith (V.zipWith (+)) g (V.map (V.map (r *)) g'))
>                 (V.zipWith (+) h (V.map (r *) h'))

> data Expr = Zero | One | Plus Expr Expr | Times Expr Expr

> instance Show Expr where
>     show Zero = "0"
>     show One = "1"
>     show (Plus a b) = "(" ++ show a ++ "+" ++ show b ++ ")"
>     show (Times a b) = "(" ++ show a ++ "*" ++ show b ++ ")"

> relu :: (Num a, Ord a) => a -> a
> relu x | x < 0 = 0
> relu x = x

> activation :: (Floating a, Ord a) => a -> a
> activation x = relu x

> run :: (Floating b, Ord b) => ArithNet b -> String -> V.Vector b -> V.Vector b
> run net "" v = v
> run net ('(' : as) v = run net as (V.map activation $ multMV (lparen net) v)
> run net (')' : as) v = run net as (V.map activation $ multMV (rparen net) v)
> run net ('+' : as) v = run net as (V.map activation $ multMV (plus net) v)
> run net ('*' : as) v = run net as (V.map activation $ multMV (times net) v)
> run net ('0' : as) v = run net as (V.map activation $ multMV (zero net) v)
> run net ('1' : as) v = run net as (V.map activation $ multMV (one net) v)

> full :: (Floating a, Ord a) => ArithNet a -> String -> a
> full net expr = dot (fini net) (run net expr (init net))

> learn :: (Expr -> Bool) -> Double -> Int -> (Expr -> Double) ->
>          Int -> ArithNet Double -> StateT StdGen IO ()
> learn _ _ _ _ 0 net = do
>     liftIO $ print $ full net "0"
>     liftIO $ print $ full net "1"

> learn cond learningRate size eval n net = do
>     expr <- randomE size
>     if cond expr
>       then do
>        when (n `mod` 1000==0) $ liftIO $ print n
>        let value = eval expr
>        let flat = show expr
>        let value' = full net flat -- Could be smarter.
>        when (n `mod` 1000==0) $
>           liftIO $ putStrLn $ show expr ++ " " ++ show value ++ " " ++ show value'
>        let g = grad (flip full flat) net
>        let net' = update net (learningRate*(value-value')) g
>        net' `deepseq` learn cond learningRate size eval (n-1) net'
>       else learn cond learningRate size eval n net

> test :: (Expr -> Bool) -> Int -> (Expr -> Double) ->
>          Int -> ArithNet Double -> StateT StdGen IO ()
> test _ _ _ 0 net = return ()

> test cond size eval n net = do
>     expr <- randomE size
>     if cond expr
>       then do
>        let value = eval expr
>        let flat = show expr
>        let value' = full net flat -- Could be smarter.
>        liftIO $ putStrLn $ show expr ++ " " ++ show value ++ " " ++ show value'
>        test cond size eval (n-1) net
>       else test cond size eval n net

> randomE :: (RandomGen g, MonadState g m) => Int -> m Expr
> randomE 1 = do
>     digit <- state $ randomR (0, 1 :: Int)
>     case digit of
>         0 -> return Zero
>         1 -> return One

> randomE size = do
>     childSize <- state $ randomR (1, size-1)
>     a <- randomE childSize
>     b <- randomE (size-childSize)
>     op <- state $ randomR (0, 1 :: Int)
>     case op of
>         0 -> return $ Plus a b
>         1 -> return $ Times a b

> num :: Int -> Int
> num 1 = 2
> num size = sum [2*num i*num (size-i) | i <- [1..(size-1)]]

> eval1 :: Num a => Expr -> a
> eval1 Zero = 0
> eval1 One = 1
> eval1 (Plus a b) = eval1 a + eval1 b
> eval1 (Times a b) = eval1 a * eval1 b

> eval2 :: Fractional a => Expr -> a
> eval2 Zero = 1.2
> eval2 One = 0.3
> eval2 (Plus a b) = 1.2*eval2 a + 0.7*eval2 b
> eval2 (Times a b) = eval2 a / (eval2 b+1)

> eval3 :: Floating a => Expr -> a
> eval3 Zero = 1.2
> eval3 One = 0.3
> eval3 (Plus a b) = tanh (eval3 a + 2.0*eval3 b-1.0)
> eval3 (Times a b) = tanh (2.0*eval3 a + eval3 b-2.0)

> main :: IO ()
> main = do
>     let dimension = 16 :: Int
>     let expSize = 6 :: Int
>     let learningRate = 0.0005
>     putStrLn $ "There are " ++ show (num expSize) ++
>                " expressions of size " ++ show expSize
>     let gen = mkStdGen 99
>     flip evalStateT gen $ do
>         net <- randomArithNet dimension
>         learn (even . asWord64 . hash . show) learningRate expSize eval1 1000000 net
>         liftIO $ print "Testing training set"
>         test (even . asWord64 . hash . show) expSize eval1 100 net
>         liftIO $ print "Testing test set"
>         test (odd . asWord64 . hash . show) expSize eval1 100 net
