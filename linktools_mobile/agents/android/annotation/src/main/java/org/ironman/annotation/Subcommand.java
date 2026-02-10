package org.ironman.annotation;

import static java.lang.annotation.ElementType.TYPE;

import java.lang.annotation.Inherited;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Retention(RetentionPolicy.CLASS)
@Target({ TYPE })
@Inherited
public @interface Subcommand {

    int order() default Integer.MAX_VALUE;

}
