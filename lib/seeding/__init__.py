from typing import Literal

MEMPTY: Literal[0] = 0
'''
The identity element of the monoid that is the sum operation over the set of integers. In other words, it is the result of `getSum mempty`.

A type `a` is a monoid if it provides an associative function that lets you combine any two values of type `a` into one, and a neutral element (`mempty`) such that

```haskell
a <> mempty == mempty <> a == a
```

A monoid is a semigroup with the added requirement of a neutral element. Therefore, any monoid is a semigroup, but not the other way around.
'''

BEST_INTEGER: Literal[3407] = 3407
'''The best integer for seeding, as proposed in https://arxiv.org/abs/2109.08203.'''

THE_ANSWER: Literal[42] = 42
'''
The answer to the ultimate question of life, the universe, and everything, established in Douglas Adams' book "The Hitchhiker's Guide to the Galaxy".
'''

BUDDHA: str = r'''
                  _oo0oo_
                 o8888888o
                 88" . "88
                 (| -_- |)
                 0\  =  /0
               ___/`---'\___
             .'   |     |   '.
            /   |||  :  ||| \
           / _||||| -:- |||||- \
          |   | \\\  -  /// |   |
          | \_|  ''\---/''  |_/ |
          \  .-\__  '-'  ___/-. /
        ___'. .'  /--.--\  `. .'___
     ."" '<  `.___\_<|>_/___.' >' "".
    | | :  `- \`.;`\ _ /`;.`/ - ` : | |
    \  \ `_.   \_ __\ /__ _/   .-` /  /
=====`-.____`.___ \_____/___.-`___.-'=====
                  `=---='
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
         佛祖保佑         永無 BUG
'''
'''The "May Buddha bless us: no bugs forever" ASCII art. Placing this ASCII art in the codebase is a common practice to prevent bugs and avoid having to debug the code.'''

HASHED_BUDDHA: Literal[3516281645] = 3516281645  # hash(BUDDHA) % 2**32
'''The hashed value of the `BUDDHA` string.'''
