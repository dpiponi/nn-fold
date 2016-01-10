My goal is to take expressions like (1+0)*(1+0+(1*0))+(1*1) and
figure out how to evaluate them based only on the values of examples.
The catch is we know that entire expressions take real values, and
that expressions are computed recursively, and that's about it.
And we know that + and * are binary operators, but we don't know
what. If we get to see examples of expressions like isolated 0's
and 1's then the problem becomes a lot easier. So to make things
hard, all examples are going to have at least 8 leaves.  This doesn't
seem like an obvious candidate for a solution via a neural network
but that's what I'm trying. Sort of.

> {-# LANGUAGE DeriveFunctor, DeriveFoldable, DeriveTraversable #-}
> {-# LANGUAGE NoMonomorphismRestriction, DeriveGeneric #-}

> import System.Random
> import Control.Monad.State
> import Control.DeepSeq
> import Control.Applicative

> import qualified GHC.Generics as G
> import qualified Data.Foldable as F
> import qualified Data.Traversable as T
> import qualified Data.Vector as V

Reverse mode automatic differentiation gives you the back-propagation
algorithm "for free".

> import Numeric.AD.Mode.Reverse as R

Some elementary linear algebra.

> dot :: Num a => V.Vector a -> V.Vector a -> a
> dot u v = V.sum $ V.zipWith (*) u v

> multMV :: Num a => V.Vector (V.Vector a) -> V.Vector a -> V.Vector a
> multMV a x = V.fromList [dot r x | r <- V.toList a]

The type ArithNet is used to represent the weights in our net. zero
represents the 'value' of the Zero symbol, times is a matrix which you
can think of as a fully connected layer designed to represent the action
of Times and so on.

> data ArithNet a = A { zero :: V.Vector a
>                     , one :: V.Vector a
>                     , plus :: V.Vector (V.Vector a)
>                     , times :: V.Vector (V.Vector a)
>                     , observe :: V.Vector a
>                     } deriving (Show, Functor, F.Foldable,
>                                 T.Traversable, G.Generic)

> instance NFData a => NFData (ArithNet a)

At the start we need to randomly initialize the weights:

> randomVector :: (Functor m, RandomGen g, MonadState g m) =>
>                 Double -> Int -> m (V.Vector Double)
> randomVector r n = fmap V.fromList $ replicateM n $
>                       state (randomR (-r, r))

> randomMatrix :: (Functor m, RandomGen g, MonadState g m) =>
>                 Double -> Int -> Int -> m (V.Vector (V.Vector Double))
> randomMatrix r m n = fmap V.fromList $ replicateM m (randomVector r n)

> randomArithNet :: (Applicative m, Functor m, RandomGen g, MonadState g m) =>
>                   Int -> m (ArithNet Double)

> randomArithNet d = A <$> randomVector 0.5 d
>                      <*> randomVector 0.5 d
>                      <*> randomMatrix (1/sqrt (fromIntegral d)) d (2*d)
>                      <*> randomMatrix (1/sqrt (fromIntegral d)) d (2*d)
>                      <*> randomVector 0.5 d

This is how we update the weights from the back-propagated gradient:

> update :: Num a => ArithNet a -> a -> ArithNet a -> ArithNet a
> update (A u v w x y) r (A u' v' w' x' y') =
>               A (V.zipWith (+) u (V.map (r *) u'))
>                 (V.zipWith (+) v (V.map (r *) v'))
>                 (V.zipWith (V.zipWith (+)) w (V.map (V.map (r *)) w'))
>                 (V.zipWith (V.zipWith (+)) x (V.map (V.map (r *)) x'))
>                 (V.zipWith (+) y (V.map (r *) y'))

The Expr type is used to represent our expressions:

> data Expr = Zero | One | Plus Expr Expr | Times Expr Expr

> instance Show Expr where
>     show Zero = "0"
>     show One = "1"
>     show (Plus a b) = "(" ++ show a ++ "+" ++ show b ++ ")"
>     show (Times a b) = "(" ++ show a ++ "*" ++ show b ++ ")"

(You can try replacing relu by tanh in the activation function.)

> relu :: (Num a, Ord a) => a -> a
> relu x | x < 0 = 0
> relu x = x

> activation :: (Floating a, Ord a) => a -> a
> activation x = relu x

Perform feed-forward evaluation (and implicitly backward propagation)
through the network.

Here's the important bit: the actual network evaluated for each
expression is derived from the form of the expression being evaluated.
The diagrams illustrate an example of how an expression is interpreted
as a network. In the second diagram the thick arrows represent
n-tuples of activation values and the boxes are linear operations,
sometimes followed by an activation function.

So the network isn't fixed and different layers in
the network may share weights.  Nonetheless the weights are still
being updated by back-propagation and the forward evaluation is the
usual stuff: linear maps followed by activation functions.  This
is a bit like a recurrent neural network except that we have a tree
structure rather than a linear structure.

> run :: (Floating b, Ord b) => ArithNet b -> Expr -> V.Vector b
> run net Zero = zero net
> run net One = one net
> run net (Plus x y) = V.map activation $
>                   multMV (plus net) (run net x V.++ run net y) 
> run net (Times x y) = V.map activation $
>                   multMV (times net) (run net x V.++ run net y) 

The final step in evaluation is an observation that converts neural net
values into a numerical value:

> full :: (Floating a, Ord a) => ArithNet a -> Expr -> a
> full net expr = dot (observe net) (run net expr)

Standard gradient descent with a fixed learning rate:

> learn :: Double -> Int -> (Expr -> Double) ->
>          Int -> ArithNet Double -> StateT StdGen IO ()
> learn _ _ _ 0 net = do
>     liftIO $ print $ full net Zero
>     liftIO $ print $ full net One

> learn learningRate size eval n net = do
>     when (n `mod` 1000==0) $ liftIO $ print n
>     expr <- randomE size
>     let value = eval expr
>     let value' = full net expr -- Could be smarter.
>     when (n < 100 || n `mod` 1000==0) $
>        liftIO $ putStrLn $ show expr ++ " " ++ show value ++ " " ++ show value'
>     let g = grad (flip full expr) net
>     let net' = update net (learningRate*(value-value')) g
>     net' `deepseq` learn learningRate size eval (n-1) net'

For training we just generate random expressions with a given number
of leaves:

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

I think this correctly computes the number of possible expressions with
a given number of leaves:

> num :: Int -> Int
> num 1 = 2
> num size = sum [2*num i*num (size-i) | i <- [1..(size-1)]]

Initially we can test with the usual interpretation of the symbols:

> eval1 :: Num a => Expr -> a
> eval1 Zero = 0
> eval1 One = 1
> eval1 (Plus a b) = eval1 a + eval1 b
> eval1 (Times a b) = eval1 a * eval1 b

We can try some bizarre semantics for our symbols:

> eval2 :: Fractional a => Expr -> a
> eval2 Zero = 1.2
> eval2 One = 0.3
> eval2 (Plus a b) = 1.2*eval2 a + 0.7*eval2 b
> eval2 (Times a b) = eval2 a / (eval2 b+1)

This serves as a regression test as we know it can be represented
(almost) exactly when using tanh as the activation function:

> eval3 :: Floating a => Expr -> a
> eval3 Zero = 1.2
> eval3 One = 0.3
> eval3 (Plus a b) = tanh (eval3 a + 2.0*eval3 b-1.0)
> eval3 (Times a b) = tanh (2.0*eval3 a + eval3 b-2.0)

It seems to work:

> main :: IO ()
> main = do
>     let dimension = 8 :: Int
>     let expSize = 8 :: Int
>     let learningRate = 0.001
>     putStrLn $ "There are " ++ show (num expSize) ++
>                " expressions of size " ++ show expSize
>     let gen = mkStdGen 99
>     flip evalStateT gen $ do
>         net <- randomArithNet dimension
>         learn learningRate expSize eval1 500000 net

If you build the code you'll see it does eventually evaluate
expressions to around 10% accuracy. For example here are the last
10 evaluations with the settings above:

((((0+1)*1)*1)+((1*0)*(1+1))) 1.0 1.085190738188251
((0+1)+(0+((0*1)+(1+(0*0))))) 2.0 2.047822955390821
(((1+((0*((0+1)*0))+1))+0)*1) 2.0 2.310788977561656
(((1+1)+0)*(((0+0)*1)+(0+1))) 2.0 2.364846216824605
((((0+1)+(0*(1+1)))+(0*0))+1) 2.0 1.9511436759887728
((((1*1)*1)*(1+1))+(1+(0*1))) 3.0 3.3042120733230633
(((1+(0*1))+(1+(0+1)))+(1+0)) 4.0 3.99495129733
(((1+1)*(0+((1+0)*1)))*(1+0)) 2.0 2.2628863627769924
(((0+(0*(0+1)))+((1*1)+0))*0) 0.0 2.498359802964492e-2
(((0+0)+(0+(0+0)))*(0*(1*1))) 0.0 -3.0017865824681578e-2

Try replacing `eval1` with `eval2` or `eval3` or using tanh instead of
relu.

It's easy to make small changes that break things. For example if
you make `eval2 One` equal -0.3 then `Times` allows you to construct
expressions with a small denominator and hence make wildly varying
values. It's hard to learn in this case.

Although the only operations available to the network are linear
mixing followed by a non-linear activation function, the weights
in the vectors zero, one and observe define a way to embed the real
line in an n-dimensional space so the net can try to find regions
in such a space where the binary operations can both be approximated.

I don't think this is the best way to solve this problem. I think
it might work better to simply build tables for the binary operators
and use spline interpolation to do lookups. You can then use something
like the Levenberg-Marquardt to fit the tables. Nonetheless, my
goal here was to see if I could fuse conventional programming (in
this case folding over a data structure) with neural networks to
see if anything useful might come of it. This approach might work
for any kind of semantics where the 'meaning' can be represented
by a point in an n-dimensional space. An example might be sentiment
analysis of parsed natural language sentences.
