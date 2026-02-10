package org.ironman.processor;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameters;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.ProcessingEnvironment;
import javax.annotation.processing.RoundEnvironment;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.tools.Diagnostic;
import org.ironman.annotation.Subcommand;

public class CommandsProcessor extends AbstractProcessor {

    @Override
    public boolean process(Set<? extends TypeElement> set, RoundEnvironment roundEnvironment) {
        final String packageName = "android.tools.processor";
        final String className = "CommandUtils";

        Set<? extends Element> elements = roundEnvironment.getElementsAnnotatedWith(Subcommand.class);
        if (elements == null || elements.isEmpty()) {
            return true;
        }
        Element[] commands = elements.toArray(new Element[0]);
        Arrays.sort(commands, (e1, e2) -> {
            Subcommand s1 = e1.getAnnotation(Subcommand.class);
            Subcommand s2 = e2.getAnnotation(Subcommand.class);
            if (s1.order() < s2.order()) {
                return -1;
            } else if (s1.order() > s2.order()) {
                return 1;
            }
            return 0;
        });

        MethodSpec.Builder method = MethodSpec.methodBuilder("addCommands")
                .addModifiers(Modifier.PUBLIC, Modifier.STATIC)
                .returns(void.class)
                .addParameter(JCommander.class, "commander");

        for (Element command : commands) {
            method.addStatement("commander.addCommand(new $T())", command.asType());
        }

        TypeSpec type = TypeSpec.classBuilder(className)
                .addModifiers(Modifier.PUBLIC, Modifier.FINAL)
                .addMethod(method.build())
                .build();

        try {
            JavaFile javaFile = JavaFile.builder(packageName, type)
                    .addFileComment(" This codes are generated automatically. Do not modify!")
                    .build();
            processingEnv.getMessager().printMessage(Diagnostic.Kind.NOTE, javaFile.toString());
            javaFile.writeTo(processingEnv.getFiler());
        } catch (IOException e) {
            processingEnv.getMessager().printMessage(Diagnostic.Kind.WARNING, e.getMessage());
        }

        return true;
    }

    @Override
    public synchronized void init(ProcessingEnvironment processingEnv) {
        super.init(processingEnv);
    }

    @Override
    public SourceVersion getSupportedSourceVersion() {
        return SourceVersion.latestSupported();
    }

    @Override
    public Set<String> getSupportedAnnotationTypes() {
        HashSet<String> set = new HashSet<>();
        set.add(Parameters.class.getCanonicalName());
        return set;
    }
}
