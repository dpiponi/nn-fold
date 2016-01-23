> {-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
> {-# LANGUAGE NoMonomorphismRestriction, DeriveGeneric #-}

This code continues from what I wrote about in
https://github.com/dpiponi/nn-fold/blob/master/test.lhs
so I'm only commenting on the differences.

In the previous code I used what I believe is called a recursive
neural net to evaluate expressions represented as tree structures.
This time the goal is more ambitious: to train a neural net to
evaluate similar expressions once they have been flattened to
strings. So the network receives as input a string like
"1+(1+1)*(1+1)+0" and must both parse and evaluate it.

The method is similar to what I used previously: carry out a fold
operation over the structure using an appropriate linear operation
(possibly followed by an activation function) for each possible
constructor in the tree. For strings that simply means folding
through the characters one by one making this something like a
recurrent neural net.

The simplest automaton for evaluating expressions like these
requires a few states as well as a stack. The stack doesn't just
hold partial evaluations, it also needs to store pending binary
operations which are only evaluated when a ')' or the end of the
string is found. But this neural net starts knowing nothing about
what any of the characters '0', '1', '+', '*', '(', and ')' mean.
For all it knows '1' is a control flow operation and '(' is a
ternary operator.

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

This is the neural net structure that has one linear operation
for each of the characters that can appear in our strings.

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

This is the fold operation on our strings that works from left-to-right.
I'm also using `init` and `fini` to signal the start and end of our strings.

> run :: (Floating b, Ord b) => ArithNet b -> String -> V.Vector b -> V.Vector b
> run _ "" v = v
> run net ('(' : as) v = run net as (V.map activation $ multMV (lparen net) v)
> run net (')' : as) v = run net as (V.map activation $ multMV (rparen net) v)
> run net ('+' : as) v = run net as (V.map activation $ multMV (plus net) v)
> run net ('*' : as) v = run net as (V.map activation $ multMV (times net) v)
> run net ('0' : as) v = run net as (V.map activation $ multMV (zero net) v)
> run net ('1' : as) v = run net as (V.map activation $ multMV (one net) v)

> full :: (Floating a, Ord a) => ArithNet a -> String -> a
> full net expr = dot (fini net) (run net expr (init net))

Learning is similar to before. However more training is required.
This means the neural net is likely to see all possible expressions
during training. Although I think it's implausible this simple
network could overlearn, i.e. cheat by simply memorizing. I don't
want this to be a possibility so the training function takes a
predicate that limits whch expressions it sees. I'm simply going
to hash all expressions to integers. The ones with even hashes will
be used for training and the ones with odd hashes will be used for
testing.

> learn :: (Expr -> Bool) -> Double -> Int -> (Expr -> Double) ->
>          Int -> ArithNet Double -> StateT StdGen IO (ArithNet Double)
> learn _ _ _ _ 0 net = return net

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
> test _ _ _ 0 _ = return ()

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

I'm using eval2 this time and I'm allowing a million examples for
training.  This is a hard problem so I'm using expressions with
just six leaves.  Nonetheless, the code has no a priori knowledge
about what the symbols mean and using six leaves means that it can't
ever use a simple reductionist approach where it deduces the value
of long expressions from the values of short ones. The value of an
expression is non-local in the sense that changing a symbol in one
place changes how a symbol at the other end gets used. So is there
any chance of this working?

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
>         net' <- learn (even . asWord64 . hash . show) learningRate expSize eval2 1000000 net
>         liftIO $ print "Testing training set"
>         test (even . asWord64 . hash . show) expSize eval2 100 net'
>         liftIO $ print "Testing test set"
>         test (odd . asWord64 . hash . show) expSize eval2 100 net'


Astonishingly it does seem to eventually get the hang of things.
Here are the last few results showing the correct value and predicted
value for the test test:

(((0+(1+0))*1)*(1*0)) 1.5433846153846154 1.3729192091351763
(((1*0)+1)*((0*0)+0)) 0.14978134110787172 0.39940289921506567
((((1+1)+0)+1)+(1*1)) 2.6080984615384617 2.580353202826156
(1+((0+0)+((1+0)+1))) 3.0836999999999994 3.174577671060543
(0+((1*(1*1))*(1+0))) 1.5175568181818182 1.0861298151963474
((1+((0*0)+0))*(0*0)) 0.9098823529411764 0.9358783459992261
(0+((0+((1+0)+1))+0)) 4.207799999999999 4.303881307367155
((0+(1+0))+(1*(1+1))) 2.8697579617834394 2.4878681737898445
(1*((0*((1*1)+1))+0)) 0.10682076069497044 0.12190171009082634
((0*(1+0))*(1+(1*1))) 0.3584888316940895 0.5040095219205387
((1+(0*1))*(0+(0*1))) 0.3260219341974077 0.34097459887588233

Remember, it's using `eval2` so the interpretations of '0', '1',
'*' and '+' are non-standard for this example. It works well with
`eval1` too.

See 
https://github.com/dpiponi/nn-fold/blob/master/nn3.png
for a graph showing the last 100 results.
